
from torch import nn


class Conv6(nn.Module):
    def __init__(self, in_channels, classes):
        super(Conv6, self).__init__()
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )


        self.flatten = nn.Flatten()

        
        self.linear = nn.Sequential(
            nn.Linear(256*4*4, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, classes,bias=True),
        )

    def forward(self, x):
        out = self.convs1(x)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    


