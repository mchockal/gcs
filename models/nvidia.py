import torch
import torch.nn as nn

class NvidiaDaveCNN(nn.Module):
    def __init__(self):
        super(NvidiaDaveCNN, self).__init__()

        self.model = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.Dropout(0.4),
            nn.Conv2d(3, 24, 5, stride=2, padding=0),
            nn.ELU(),
            nn.LazyConv2d(36, 5, stride=2, padding=0),
            nn.ELU(),
            nn.LazyConv2d(48, 5, stride=2, padding=0),
            nn.ELU(),
            nn.LazyConv2d(64, 3, stride=1, padding=0),
            nn.ELU(),
            nn.LazyConv2d(64, 3, stride=1, padding=0),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.LazyLinear(out_features=104),
            nn.ELU(),
            nn.LazyLinear(out_features=56),
            nn.ELU(),
            nn.LazyLinear(out_features=8),
            nn.ELU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, x):
        return self.model(x).double()
