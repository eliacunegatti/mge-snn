
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

def get_optimizer(model, dataset, m):
    if dataset == 'CIFAR-10' or dataset == 'CIFAR-100':
        epochs = 160
        if m == 'CONV-4':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=1e-4) # 
        elif m == 'CONV-6':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=1e-4) # 
        elif m == 'Resnet-20' or m == 'Resnet-32':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=1e-4) # 
        elif  m == 'Wide-Resnet-28-2':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9 ,weight_decay=5e-4)
        elif m == 'MobileNetV2':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9 ,weight_decay=1e-4)

    elif dataset == 'tinyimagenet':
        epochs = 100
        if m == 'CONV-4' or m == 'CONV-6':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=1e-4) # 
        elif m == 'Resnet-20' or m == 'Resnet-32':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=1e-4) # 
        elif  m == 'Wide-Resnet-28-2':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9, weight_decay=5e-4)
        elif m == 'MobileNetV2':
            optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9 ,weight_decay=1e-4)
            
    return optimizer, epochs

def get_scheduler(optimizer, dataset, m):
    if dataset == 'CIFAR-10' or dataset == 'CIFAR-100':
        epochs = 160
        if m == 'CONV-4':
            scheduler = MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
        elif m == 'CONV-6':
            scheduler = MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
        elif m == 'Resnet-20' or m == 'Resnet-32' or m == 'Wide-Resnet-28-2' or m == 'MobileNetV2':
            scheduler = MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
    elif dataset == 'tinyimagenet':
        if m == 'CONV-4':
            scheduler = MultiStepLR(optimizer, milestones=[20,60,80], gamma=0.1)
        elif m == 'CONV-6':
            scheduler = MultiStepLR(optimizer, milestones=[20,60,80], gamma=0.1)
        else:
            scheduler = MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.1)      

    return scheduler


