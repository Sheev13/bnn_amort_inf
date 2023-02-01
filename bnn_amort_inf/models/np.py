from typing import Dict, List, Tuple

import torch
from torch import nn

from ..utils.activations import NormalActivation
from ..utils.networks import CNN, MLP, SetConv


class CNPEncoder(nn.Module):
    """Represents the deterministic encoder for a conditional neural process."""

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

        return self.mlp(torch.cat((x_c, y_c), dim=-1)).sum(0)


class ConvCNPEncoder(nn.Module):
    """Represents the encoder of a convolutional conditional neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cnn_dims: List[int],
        kernel_size: int,
        granularity: int,
        conv: nn.Module = nn.Conv1d,
        nonlinearity: nn.Module = nn.ReLU(),
        normalisation: nn.Module = nn.Identity,  # nn.BatchNorm1d
        **conv_layer_kwargs,
    ):
        super().__init__()

        if cnn_dims[-1] != 2:
            cnn_dims.append(2)  # output channels for mu and sigma

        if cnn_dims[0] != y_dim:
            cnn_dims = [y_dim] + cnn_dims

        self.cnn = CNN(
            cnn_dims,
            kernel_size,
            conv,
            nonlinearity,
            normalisation,
            **conv_layer_kwargs,
        )

        self.set_conv = SetConv(
            x_dim, y_dim, train_lengthscale=True, lengthscale=0.05 * granularity
        )

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.granularity = granularity
        self.nonlinearity = nonlinearity
        self.embedded_dim = cnn_dims[-1]

    def forward(self, x_c, y_c, x_t):
        assert len(x_c.shape) == 2
        assert len(x_t.shape) == 2
        assert len(y_c.shape) == 2
        assert x_c.shape[0] == y_c.shape[0]
        assert x_c.shape[1] == self.x_dim
        assert x_t.shape[1] == self.x_dim
        assert y_c.shape[1] == self.y_dim

        x_min = min(torch.min(x_c), torch.min(x_t)) - 0.1
        x_max = max(torch.max(x_c), torch.max(x_t)) + 0.1
        num_points = int((x_max - x_min) * self.granularity)
        x_grid = torch.linspace(x_min, x_max, num_points)

        F = self.set_conv(x_c, y_c, x_grid).T
        F = self.nonlinearity(F)
        F = self.cnn(F.unsqueeze(0)).T.squeeze()

        assert len(F.shape) == 2
        assert F.shape[0] == num_points
        assert F.shape[1] == 2

        return (F, x_grid)


class CNPDecoder(nn.Module):
    """Represents the decoder for a conditional neural process."""

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


class ConvCNPDecoder(nn.Module):
    """Represents the decoder of a convolutional conditional neural process."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        granularity: int,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mean_layer = SetConv(
            1, y_dim, train_lengthscale=True, lengthscale=2.0 / granularity
        )

        self.log_sigma_layer = SetConv(
            1, y_dim, train_lengthscale=True, lengthscale=2.0 / granularity
        )

    def forward(
        self,
        E: Tuple[torch.Tensor, torch.Tensor],
        x_t: torch.Tensor,
    ) -> torch.distributions.Distribution:
        z_c, x_grid = E
        assert len(x_t.shape) == 2
        assert x_t.shape[1] == self.x_dim

        mean = self.mean_layer(x_grid.unsqueeze(1), z_c[:, 0].unsqueeze(1), x_t)
        sigma = self.log_sigma_layer(
            x_grid.unsqueeze(1), z_c[:, 1].unsqueeze(1), x_t
        ).exp()
        return torch.distributions.Normal(mean, sigma)


class BaseNP(nn.Module):
    """Represents a neural process base class"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        nonlinearity: nn.Module,
        noise: float = 1e-2,
        train_noise: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.nonlinearity = nonlinearity

        self.encoder = None
        self.decoder = None

        self.log_noise = nn.Parameter(
            torch.tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    @property
    def conv(self):
        return isinstance(self.encoder, ConvCNPEncoder)

    def forward(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        if self.encoder is None or self.decoder is None:
            raise NotImplementedError

        if self.conv:
            decode = self.decoder(self.encoder(x_c, y_c, x_t), x_t)
        else:
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
        noise: float = 1e-2,
        train_noise: bool = True,
        decoder_activation: nn.Module = NormalActivation(),
    ):
        super().__init__(
            x_dim,
            y_dim,
            nonlinearity,
            noise,
            train_noise,
        )

        self.encoder = CNPEncoder(
            x_dim,
            y_dim,
            encoder_hidden_dims,
            embedded_dim,
            nonlinearity=nonlinearity,
            activation=nn.Identity(),
        )

        self.decoder = CNPDecoder(
            x_dim,
            y_dim,
            decoder_hidden_dims,
            embedded_dim,
            decoder_activation,
            nonlinearity,
        )


class ConvCNP(BaseNP):
    """Represents a Convolutional Conditional Neural Process"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cnn_dims: List[int] = [64, 64],
        kernel_size: int = 8,
        granularity: int = 64,  # discretized points per unit
        nonlinearity: nn.Module = nn.ReLU(),
        **conv_layer_kwargs,
    ):
        super().__init__(
            x_dim,
            y_dim,
            nonlinearity,
        )

        self.encoder = ConvCNPEncoder(
            x_dim,
            y_dim,
            cnn_dims,
            kernel_size,
            granularity,
            conv=nn.Conv1d,
            nonlinearity=nn.ReLU(),
            normalisation=nn.Identity,  # nn.BatchNorm1d
            **conv_layer_kwargs,
        )

        self.decoder = ConvCNPDecoder(
            x_dim,
            y_dim,
            granularity,
        )
