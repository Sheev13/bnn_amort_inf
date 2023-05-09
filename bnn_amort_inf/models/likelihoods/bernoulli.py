import torch

from .base import Likelihood


class BernoulliLikelihood(Likelihood):
    def forward(self, x: torch.Tensor) -> torch.distributions.Bernoulli:
        return torch.distributions.Bernoulli(logits=x)
