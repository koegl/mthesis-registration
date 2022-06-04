# from https://github.com/lucidrains/vit-pytorch

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def triple(t):
    return t if isinstance(t, tuple) else (t, t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, attention_or_mlp):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attention_or_mlp = attention_or_mlp

    def forward(self, x, **kwargs):
        normalised_input = self.norm(x)
        attention_or_mlp_output = self.attention_or_mlp(normalised_input, **kwargs)

        return attention_or_mlp_output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # temp
        self.dim = dim
        self.inner_dimension = inner_dim

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTStandard3D(nn.Module):
    def __init__(self, *, volume_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=2, dim_head=64, dropout=0., emb_dropout=0., device="cpu"):
        super().__init__()

        self.device = device

        volume_height, volume_width, volume_depth = triple(volume_size)
        patch_height, patch_width, patch_depth = triple(patch_size)

        assert volume_height % patch_height == 0 and \
               volume_width % patch_width == 0 and \
               volume_depth % patch_depth == 0, \
               'Image dimensions must be divisible by the patch size.'

        num_patches = (volume_height // patch_height) * (volume_width // patch_width) * (volume_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
                      p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, volume):

        # convert input image into patch embeddings
        x = self.to_patch_embedding(volume)
        b, n, dim = x.shape

        # get the cls token and repeat it for the amount of elements in the batch
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)

        # add the positional embeddings (it has shape (1, n+1, dim) but it gets added to each element of the batch
        x += self.pos_embedding[:, :(n + 1)]

        # apply dropout
        x = self.dropout(x)

        # feed the embeddings through the transformer
        x = self.transformer(x)

        # for classification we extract only the first 'path,' which is the classification token
        x = x[:, 0]

        x = self.mlp_head(x)

        # x = self.softmax(x)
        # x = self.sigmoid(x)

        return x
