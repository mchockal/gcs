import torch
from torchvision import models
from torchsummary import summary

vit = models.VisionTransformer(
    image_size=128, patch_size=64, num_layers=2, num_heads=2, hidden_dim=64, mlp_dim=1024,
).to('cpu')

summary(vit, (3, 128, 128))
