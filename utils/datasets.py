import torch
from torchvision import datasets, transforms
from tinyimagenet.tiny_imagenet import TinyImageNet
import copy
def get_dataset(dataset):
    if dataset == 'CIFAR-10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

        trainset = datasets.CIFAR10('dataset/', download=True, train=True, transform=transform_train)
        testset = datasets.CIFAR10('dataset/', download=True, train=False, transform=transform_test)
        n = 32
        channels = 3
        classes = 10
        trainloader = torch.utils.data.DataLoader(trainset, 128, shuffle=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, 128, shuffle=False, num_workers=4, pin_memory=True)
    
    elif dataset == 'CIFAR-100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        transform_test = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        trainset = datasets.CIFAR100('dataset/', download=True, train=True, transform=transform_train)
        testset = datasets.CIFAR100('dataset/', download=True, train=False, transform=transform_test)
        n = 32
        channels = 3
        classes = 100
        trainloader = torch.utils.data.DataLoader(trainset, 128, shuffle=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, 128, shuffle=False,num_workers=4, pin_memory=True)
    
    elif dataset == 'tinyimagenet':
        batch_size = 128
        C = TinyImageNet(batch_size)
        trainloader = copy.deepcopy(C.train_loader)
        testloader = copy.deepcopy(C.val_loader)
        del C
        channels = 3
        classes = 200
        n = 32
    else:
        raise Exception('Dataset not available')

    return trainloader, testloader, n, channels, classes