from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class Pruner(ABC):
    """Pruner abstract class."""

    def __init__(
        self, net: nn.Module, device: torch.device, input_shape: List[int]
    ) -> None:
        """Initialize."""
        super(Pruner, self).__init__()
        self.model = net
        self.device = device
        self.input_shape = input_shape
        self.params_to_prune: Tuple[Tuple[nn.Module, str]] = None  # type: ignore

    @abstractmethod
    def prune(self, amount):
        """Prune."""
        pass

    @abstractmethod
    def get_prune_score(self):
        """Get prune score."""
        pass

    def global_unstructured(
        self, pruning_method: torch.nn.utils.prune.BasePruningMethod, **kwargs
    ):
        """Based on
        https://pytorch.org/docs/stable/_modules/torch/nn/utils/prune.html#global_unstructured.
        Modify scores depending on the algorithm.
        """
        assert isinstance(self.params_to_prune, Iterable)

        scores = self.get_prune_score()

        t = torch.nn.utils.parameters_to_vector(scores)
        # similarly, flatten the masks (if they exist), or use a flattened vector
        # of 1s of the same dimensions as t
        default_mask = torch.nn.utils.parameters_to_vector(
            [
                getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
                for (module, name) in self.params_to_prune  # type: ignore
            ]
        )

        # use the canonical pruning methods to compute the new mask, even if the
        # parameter is now a flattened out version of `parameters`
        container = prune.PruningContainer()
        container._tensor_name = "temp"  # type: ignore
        method = pruning_method(**kwargs)
        method._tensor_name = "temp"  # type: ignore
        if method.PRUNING_TYPE != "unstructured":
            raise TypeError(
                'Only "unstructured" PRUNING_TYPE supported for '
                "the `pruning_method`. Found method {} of type {}".format(
                    pruning_method, method.PRUNING_TYPE
                )
            )

        container.add_pruning_method(method)

        # use the `compute_mask` method from `PruningContainer` to combine the
        # mask computed by the new method with the pre-existing mask
        final_mask = container.compute_mask(t, default_mask)

        # Pointer for slicing the mask to match the shape of each parameter
        pointer = 0
        for module, name in self.params_to_prune:  # type: ignore
            param = getattr(module, name)
            # The length of the parameter
            num_param = param.numel()
            # Slice the mask, reshape it
            param_mask = final_mask[pointer : pointer + num_param].view_as(param)
            # Assign the correct pre-computed mask to each parameter and add it
            # to the forward_pre_hooks like any other pruning method
            prune.custom_from_mask(module, name, param_mask)

            # Increment the pointer to continue slicing the final_mask
            pointer += num_param

    def get_params(
        self, extract_conditions: Tuple[Tuple[Any, str], ...]
    ) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters(weight and bias) tuples for pruning."""
        t = []
        for module in self.model.modules():
            for module_type, param_name in extract_conditions:
                # it returns true when we try hasattr(even though it returns None)
                if (
                    isinstance(module, module_type)
                    and hasattr(module, param_name)
                    and getattr(module, param_name) is not None
                ):
                    t += [(module, param_name)]
        return tuple(t)

    def mask_sparsity(
        self,
        module_types: Tuple[Any, ...] = (
            nn.Conv2d,
            nn.Linear,
        ),
    ) -> float:
        """Get the ratio of zeros in weight masks."""
        self.mask = []
        n_zero = n_total = 0
        for module, param_name in self.params_to_prune:
            match = next((m for m in module_types if type(module) is m), None)
            if not match:
                continue
            param_mask_name = param_name + "_mask"
            if hasattr(module, param_mask_name):
                param = getattr(module, param_mask_name)
                self.mask.append(param)
                n_zero += int(torch.sum(param == 0.0).item())
                n_total += param.nelement()

        return (100.0 * n_zero / n_total) if n_total != 0 else 0.0


class Synflow(Pruner):
    def __init__(
        self, net: nn.Module, device: torch.device, input_shape: List[int]
    ) -> None:
        super(Synflow, self).__init__(net, device, input_shape)

        self.params_to_prune = self.get_params(
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias"),
            )
        )
        prune.global_unstructured(
            self.params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.0,
        )
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        # To get gradient of each weight(after prune at least one time)
        self.params_to_prune_orig = self.get_params(
            (
                (nn.Conv2d, "weight_orig"),
                (nn.Conv2d, "bias_orig"),
                (nn.Linear, "weight_orig"),
                (nn.Linear, "bias_orig"),
            )
        )

    def prune(self, amount: int):
        unit_amount = 1 - ((1 - amount) ** 0.01)
        print(f"Start prune, target_sparsity: {amount*100:.2f}%")
        for _ in range(100):
            self.global_unstructured(
                pruning_method=prune.L1Unstructured, amount=unit_amount
            )
        sparsity = self.mask_sparsity()
        print(f"Pruning Done, sparsity: {sparsity:.2f}%")

    def get_prune_score(self) -> List[float]:
        """Run prune algorithm and get score."""
        # Synaptic flow
        signs = self.linearize()
        input_ones = torch.ones([1] + self.input_shape).to(self.device)
        self.model.eval()
        output = self.model(input_ones)
        torch.sum(output).backward()

        # get score function R
        scores = []
        for (p, n), (po, no) in zip(self.params_to_prune, self.params_to_prune_orig):
            score = (getattr(p, n) * getattr(po, no).grad).to(self.device).detach().abs_()
            scores.append(score)
            getattr(po, no).grad.data.zero_()

        self.nonlinearize(signs)
        self.model.train()
        return scores

    @torch.no_grad()
    def linearize(self):
        signs = {}
        for name, param in self.model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(self, signs: Dict[str, torch.Tensor]):
        for name, param in self.model.state_dict().items():
            param.mul_(signs[name])