from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .gibnn_layers import FreeFormGIBNNLayer


class BaseGIBNN(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        noise: float = 1.0,
        train_noise: bool = False,
        amortised=False,
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
        self.amortised = amortised

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
        }
        return elbo, metrics

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        elbo, metrics = self.elbo(x, y, num_samples)
        return (-elbo / x.shape[0]), metrics

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class GIBNN(BaseGIBNN):
    """Represents the standard Global Inducing Point BNN"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        num_inducing: int,
        inducing_points: Optional[torch.Tensor] = None,
        nonlinearity: nn.Module = nn.ReLU(),
        pws: Optional[List[torch.distributions.Normal]] = None,
        noise: float = 1.0,
        train_noise: bool = False,
        amortised=False,
    ):
        super().__init__(
            x_dim, y_dim, hidden_dims, nonlinearity, noise, train_noise, amortised
        )

        self.num_inducing = num_inducing

        dims = [x_dim] + hidden_dims + [y_dim]
        if pws is None:
            pws = [None] * (len(dims) - 1)
        else:
            assert len(pws) == (len(dims) - 1)

        if inducing_points is None:
            self.inducing_points = nn.Parameter(torch.randn(num_inducing, x_dim))
        else:
            assert inducing_points.shape == torch.Size((num_inducing, x_dim))
            self.inducing_points = nn.Parameter(inducing_points)

        # ModuleList for storing GIBNNLayers.
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                FreeFormGIBNNLayer(
                    dims[i] + 1,
                    dims[i + 1],
                    num_pseudo=num_inducing,
                    pw=pws[i],
                )
            )

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert len(x.shape) == 2
        assert x.shape[-1] == self.x_dim

        # (num_samples, N, input_dim).
        F = x.unsqueeze(0).repeat(num_samples, 1, 1)
        U = self.inducing_points.unsqueeze(0).repeat(num_samples, 1, 1)

        kl_total = None
        for i, layer in enumerate(self.layers):
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)
            U_ones = torch.ones(U.shape[:-1]).unsqueeze(-1)
            U = torch.cat((U, U_ones), dim=-1)

            # Sample GIBNNLayer.
            layer(U)

            w = layer.cache["w"]
            kl = layer.cache["kl"]

            # (num_samples, batch_size, output_dim).
            F = F @ w.transpose(-1, -2)
            U = U @ w.transpose(-1, -2)
            if i < len(self.layers) - 1:
                F = self.nonlinearity(F)
                U = self.nonlinearity(U)

            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        return F, kl_total
