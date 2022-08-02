import torch
import torch.nn as nn

class NvidiaDaveCNN_2(nn.Module):
    def __init__(self):
        super(NvidiaDaveCNN_2, self).__init__()

        self.model = nn.Sequential(
            # nn.LazyBatchNorm2d(),  
            nn.LazyConv2d(20, 7, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(25, 6, stride=2, padding=0),  # 100
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(40, 5, stride=2, padding=0),  # 150
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(55, 4, stride=1, padding=0),  # 200
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(70, 3, stride=1, padding=0),  # 300
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(85, 2, stride=1, padding=0),  # 300
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.LazyLinear(out_features=104),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=56),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=8),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, x):
        return self.model(x).double()

