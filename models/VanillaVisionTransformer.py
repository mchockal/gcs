import torch
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, device, image_shape, patch_size, hidden_dim, embed_dim, num_channels, num_heads, num_layers, dropout=0.0):
        super(VisionTransformer, self).__init__()
        assert hidden_dim % num_heads == 0

        # calculates number of patches
        C, H, W = image_shape
        num_patches = (H * W) // (patch_size * patch_size)

        self.image_shape = image_shape
        self.device = device
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches

        self.linear_projection = nn.Linear(self.num_channels*(self.patch_size**2), self.embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads, dropout) for _ in range(self.num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                      nn.Linear(self.embed_dim, self.hidden_dim))

        # not sure if this final layer is necessary. Alternatively we can have the mlp_head output to 1 value.
        self.final_linear = nn.Linear(self.hidden_dim, 1)

        self.class_token = nn.Parameter(torch.rand(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1+self.num_patches, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        BATCH, NUM_PATCHES, _ = images.shape
        x = self.linear_projection(images)

        cls_token = self.class_token.repeat(BATCH, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :NUM_PATCHES+1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        out = self.mlp_head(cls)

        out = self.final_linear(out)
        return out


