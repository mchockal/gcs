import torch
import torch.nn as nn

class NvidiaDaveCNN_2(nn.Module):
    def __init__(self):
        super(NvidiaDaveCNN_2, self).__init__()

        self.model = nn.Sequential(
            # nn.LazyBatchNorm2d(),  
            nn.LazyConv2d(65, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(100, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(150, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(200, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(300, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.LazyLinear(out_features=104),
            nn.ReLU(),
            nn.LazyLinear(out_features=56),
            nn.ReLU(),
            nn.LazyLinear(out_features=8),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, x):
        return self.model(x).double()

