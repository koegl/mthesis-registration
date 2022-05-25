import torch
from torch import nn
from einops import rearrange, repeat

from time import perf_counter

from logic.vit import Attention


def func(val):
    # invert torch tensor
    return val.pow(10)


device = "mps"
device = "cpu"

time = perf_counter()
# create random torch tensor
x = torch.randn(1, 50, 128)
# x = torch.randn(10000, 10000)
x = x.to(device)

model = Attention(128, heads=8, dim_head=64, dropout=0.)

output = model(x)

print(f"Time: {perf_counter() - time}")

print(output.shape)
