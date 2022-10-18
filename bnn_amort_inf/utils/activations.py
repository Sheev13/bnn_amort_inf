import torch
from torch import nn


class NormalActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        assert x.shape[-1] % 2 == 0, "x must have a multiple of 2 dimensions."

        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        logvar = logvar - 4.0
        return torch.distributions.Normal(loc, logvar.exp().pow(0.5))


class CategoricalActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Categorical(logits=x)


class BernoulliActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Bernoulli(logits=x)
