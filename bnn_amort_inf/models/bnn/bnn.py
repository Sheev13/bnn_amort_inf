from abc import ABC
from typing import Dict, List, Tuple

import torch
from torch import nn


class BaseBNN(nn.Module, ABC):
    amortised = False

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        noise: float = 1.0,
        train_noise: bool = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity

        # Observation noise.
        self.log_noise = nn.Parameter(
            torch.tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    def exp_ll(self, F: torch.Tensor, y: torch.Tensor):
        assert len(y.shape) == 2
        assert y.shape[-1] == self.y_dim

        y = y.unsqueeze(0).repeat(F.shape[0], 1, 1)
        assert y.shape == F.shape

        # (num_samples, N, output_dim)
        log_probs = torch.distributions.normal.Normal(F, self.noise).log_prob(y)
        return log_probs.sum(-1).sum(-1)

    def elbo(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.amortised:
            F, kl = self(x, y, num_samples=num_samples)[:2]
        else:
            F, kl = self(x, num_samples=num_samples)[:2]

        exp_ll = self.exp_ll(F, y).mean(0)
        kl = kl.mean(0)
        elbo = exp_ll - kl

        metrics = {
            "elbo": elbo.item(),
            "exp_ll": exp_ll.item(),
            "kl": kl.item(),
            "noise": self.noise.item(),
        }
        return elbo, metrics

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        elbo, metrics = self.elbo(x, y, num_samples)
        return (-elbo / x.shape[0]), metrics

    def forward(self, *args, **kwargs):
        raise NotImplementedError
