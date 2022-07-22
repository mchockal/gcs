import torch
import torch.nn as nn

class NvidiaDaveCNN(nn.Module):
    def __init__(self):
        super(NvidiaDaveCNN, self).__init__()

        self.model = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.Conv2d(3, 24, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(36, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(48, 5, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1, padding=0),
            nn.ReLU(),
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
