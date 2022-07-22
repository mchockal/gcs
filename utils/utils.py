import os

import torch

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))