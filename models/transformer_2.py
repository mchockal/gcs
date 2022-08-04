import torch
import torch.nn as nn
import torchvision

class VisionTransformer(nn.Module):
    def __init__(self, 
                image_size=208, #224
                patch_size=32, 
                num_layers=12, 
                num_heads=12, 
                hidden_dim=768, 
                mlp_dim=3072):
        super(VisionTransformer, self).__init__()

        # WARNING - Do not use this as it causes memory issues
        # due to not being able to specify a smaller image size.
        #self.vit = torchvision.models.vit_b_16()

        # JTS note:  Original patch size is 16


        self.vit = torchvision.models.VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )

        self.linear = nn.Linear(in_features=1000, out_features=1)

    def forward(self, x):
        x = self.vit(x)
        return self.linear(x).double()
