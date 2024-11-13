import os
import torch
from torchvision import datasets, transforms
import torchvision
torch.multiprocessing.set_sharing_strategy("file_system")

class TinyImageNet:
    def __init__(self, batch_size):
        super(TinyImageNet, self).__init__()
        data_root = os.path.join('data', "tiny-imagenet-200")

        normalize = transforms.Normalize(
            mean=[0.48024578664982126, 0.44807218089384643, 0.3975477478649648],
            std=[0.2769864069088257, 0.26906448510256, 0.282081906210584]
        )

        # Data augmentation for training
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BOX),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(
            os.path.join(data_root, "train"),
            train_transforms,
        )

        # Validation dataset
        val_transforms = transforms.Compose([
            transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BOX),
            transforms.ToTensor(),
            normalize,
        ])

        val_dataset = datasets.ImageFolder(
            os.path.join(data_root, "val"),
            val_transforms,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        )