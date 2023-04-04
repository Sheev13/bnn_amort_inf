from typing import List

import torch
from torch import nn

from ...utils.networks import MLP
from ..likelihoods.base import Likelihood
from ..likelihoods.normal import NormalLikelihood
from .base import BaseNP


class CNPEncoder(nn.Module):
    """Represents the deterministic encoder for a conditional neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        embedded_dim: int,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim

        self.mlp = MLP(
            [x_dim + y_dim] + hidden_dims + [embedded_dim],
            nonlinearity=nonlinearity,
        )

    def forward(self, x_c: torch.Tensor, y_c: torch.Tensor) -> torch.Tensor:
        assert len(x_c.shape) == 2
        assert len(y_c.shape) == 2
        assert x_c.shape[0] == y_c.shape[0]
        assert x_c.shape[1] == self.x_dim
        assert y_c.shape[1] == self.y_dim

        return self.mlp(torch.cat((x_c, y_c), dim=-1)).sum(0)


class CNPDecoder(nn.Module):
    """Represents the decoder for a conditional neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        embedded_dim: int,
        nonlinearity: nn.Module = nn.ReLU(),
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim
        self.likelihood = likelihood

        self.mlp = MLP(
            [embedded_dim + x_dim]
            + hidden_dims
            + [self.likelihood.out_dim_multiplier * y_dim],
            nonlinearity=nonlinearity,
        )

    def forward(
        self, z_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        assert len(z_c.shape) == 1
        assert len(x_t.shape) == 2
        assert z_c.shape[0] == self.embedded_dim
        assert x_t.shape[1] == self.x_dim

        # Use same context for each prediction.
        z_c = z_c.unsqueeze(0).repeat(x_t.shape[0], 1)
        return self.likelihood(self.mlp(torch.cat((z_c, x_t), dim=-1)))


class CNP(BaseNP):
    """Represents a Conditional Neural Process"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        encoder_hidden_dims: List[int],
        embedded_dim: int,
        decoder_hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
    ):
        encoder = CNPEncoder(
            x_dim,
            y_dim,
            encoder_hidden_dims,
            embedded_dim,
            nonlinearity=nonlinearity,
        )

        decoder = CNPDecoder(
            x_dim, y_dim, decoder_hidden_dims, embedded_dim, nonlinearity, likelihood
        )
        super().__init__(x_dim, y_dim, embedded_dim, encoder, decoder)
