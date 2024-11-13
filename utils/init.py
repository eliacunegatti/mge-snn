import torch
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        try:
            torch.nn.init.zeros_(m.bias)
            #m.bias.requires_grad = False
        except:
            pass

def mask_bias(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        try:
            #torch.nn.init.zeros_(m.bias)
            m.bias.requires_grad = False
        except:
            pass