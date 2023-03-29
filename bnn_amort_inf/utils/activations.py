from typing import Optional

import torch
from torch import nn


class NormalActivation(nn.Module):
    def forward(
        self, x: torch.Tensor, dim: Optional[int] = -1
    ) -> torch.distributions.Distribution:
        assert x.shape[dim] % 2 == 0, "x must have a multiple of 2 dimensions."
        loc, logvar = torch.split(x, x.shape[dim] // 2, dim=dim)
        logvar = logvar - 4.0
        return torch.distributions.Normal(loc, (logvar.exp().pow(0.5) + 1e-8))


class CategoricalActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Categorical(logits=x)


class BernoulliActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Bernoulli(logits=x)
