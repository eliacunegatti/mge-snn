import torch
import numpy as np
from torch import nn

def apply_prune_mask(net, keep_masks, device):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask.to(device)

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))


def count_paramaters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    no_params = []
    for w in model.parameters():
        no_params.append(int(torch.count_nonzero(w)))


    r = float(sum(no_params)/params)
    print('Total Number of parameters (dense version): {0}\nTotal Number of parameters (sparse version): {1}\nDensity: {3}\nSparsity: {2}'.format(params, sum(no_params), 1-r, r))


def mask_gradient(model, keep_mask):
    for idx, t in enumerate(keep_mask):
        t = (1-t)
        keep_mask[idx] = t
    model.prune_mask = keep_mask

def mask_bias(model):
    for name, params in (model.named_parameters()):
        if params.requires_grad:
            if torch.count_nonzero(params) ==  params.numel() and len(params.shape) > 1:
                params.requires_grad = False