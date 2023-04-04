import torch
from torch import nn


class Likelihood(nn.Module):
    out_dim_multiplier = 1

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError
