from typing import Dict, List, Tuple

import torch
from torch import nn

from ..utils.activations import NormalActivation
from ..utils.networks import MLP


class Encoder(nn.Module):
    """Represents the deterministic encoder for a neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        embedded_dim: int,
        nonlinearity: nn.Module = nn.ReLU(),
        activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim

        self.mlp = MLP(
            [x_dim + y_dim] + hidden_dims + [embedded_dim],
            nonlinearity=nonlinearity,
            activation=activation,
        )

    def forward(self, x_c: torch.Tensor, y_c: torch.Tensor) -> torch.Tensor:
        assert len(x_c.shape) == 2
        assert len(y_c.shape) == 2
        assert x_c.shape[0] == y_c.shape[0]
        assert x_c.shape[1] == self.x_dim
        assert y_c.shape[1] == self.y_dim

        return self.mlp(torch.cat((x_c, y_c), dim=-1))


class CNPEncoder(Encoder):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return super().forward(*args, **kwargs).sum(0)


class Decoder(nn.Module):
    """Represents the decoder for a neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        embedded_dim: int,
        activation: nn.Module,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim

        if isinstance(activation, NormalActivation):
            self.mlp = MLP(
                [embedded_dim + x_dim] + hidden_dims + [2 * y_dim],
                nonlinearity=nonlinearity,
                activation=activation,
            )
        else:
            self.mlp = MLP(
                [embedded_dim + x_dim] + hidden_dims + [y_dim],
                nonlinearity=nonlinearity,
                activation=activation,
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
        return self.mlp(torch.cat((z_c, x_t), dim=-1))


class CNP(nn.Module):
    """Represents a vanilla conditional neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        noise: float = 1e-2,
        train_noise: bool = True,
        decoder_activation: nn.Module = NormalActivation(),
    ):
        super().__init__()

        self.encoder = CNPEncoder(
            x_dim, y_dim, encoder_hidden_dims, embedded_dim, nonlinearity
        )
        self.decoder = Decoder(
            x_dim,
            y_dim,
            decoder_hidden_dims,
            embedded_dim,
            decoder_activation,
            nonlinearity,
        )

        self.log_noise = nn.Parameter(
            torch.tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    def forward(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        decode = self.decoder(self.encoder(x_c, y_c), x_t)
        if isinstance(decode, torch.distributions.Distribution):
            return decode
        else:
            return torch.distributions.Normal(decode, self.noise)

    def loss(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor, y_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dist = self.forward(x_c, y_c, x_t)
        ll = dist.log_prob(y_t).sum()

        metrics = {"ll": ll.item(), "noise": self.noise.detach().item()}
        return (-ll / x_t.shape[0]), metrics
