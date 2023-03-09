from abc import ABC
from typing import Callable, Dict, List, Tuple

import torch
from torch import nn

from ..likelihoods.normal import NormalLikelihood


class BaseBNN(nn.Module, ABC):
    amortised = False

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        likelihood: Callable = NormalLikelihood(noise=1.0),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.likelihood = likelihood

    def elbo(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.amortised:
            F, kl = self(x, y, num_samples=num_samples)[:2]
        else:
            F, kl = self(x, num_samples=num_samples)[:2]

        qy = self.likelihood(F)
        exp_ll = (
            qy.log_prob(y.unsqueeze(0).repeat(num_samples, 1, 1)).sum() / num_samples
        )
        kl = kl.mean(0)
        elbo = exp_ll - kl

        metrics = {
            "elbo": elbo.item(),
            "exp_ll": exp_ll.item(),
            "kl": kl.item(),
            # "noise": self.noise.item(),
        }
        return elbo, metrics

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        elbo, metrics = self.elbo(x, y, num_samples)
        return (-elbo / x.shape[0]), metrics

    def forward(self, *args, **kwargs):
        raise NotImplementedError
