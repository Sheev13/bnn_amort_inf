from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.activations import NormalActivation
from ..utils.networks import CNN, MLP, SetConv, Unet


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                self.weight.exp(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


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
    """Represents a ConvCNP encoder for off-the-grid data.."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int,
        granularity: int,
        encoder_lengthscale: float = 0.1,
    ):
        super().__init__()

        self.set_conv = SetConv(
            x_dim,
            embedded_dim,
            train_lengthscale=True,
            lengthscale=encoder_lengthscale,
        )

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim
        self.granularity = granularity

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

        F = self.set_conv(x_c, y_c, x_grid).permute(1, 0)

        assert F.shape[0] == self.embedded_dim
        assert F.shape[1] == num_points

        return (F, x_grid)


class GridConvCNPEncoder(nn.Module):
    """Represents the encoder for a ConvCNP operating on-the-grid"""

    def __init__(
        self,
        x_dim: int,  # should be 2 for an image
        y_dim: int,  # should be 3 for a colour image (rgb)
        embedded_dim: int,
        conv_kernel_size: int,
        conv: nn.Module = nn.Conv2d,
    ):
        super().__init__()

        # might need to do something here to ensure positivity!
        self.conv = make_abs_conv(conv)
        self.conv = self.conv(
            y_dim,
            y_dim,
            kernel_size=conv_kernel_size,
            groups=y_dim,
            padding="same",
        )  # (y_dim)

        self.resizer = nn.Sequential(nn.Linear(y_dim * 2, embedded_dim))

        self.embedded_dim = embedded_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, I: torch.Tensor, M_c: torch.Tensor) -> torch.Tensor:
        assert len(I.shape) == self.x_dim + 1  # extra dimension for colour channels
        assert I.shape[0] == self.y_dim
        if len(M_c.shape) == self.x_dim:
            M_c = M_c.unsqueeze(0)
        assert len(M_c.shape) == len(I.shape)

        I_c = I * M_c  # get context pixels from image and context mask
        # shape (y_dim, *grid_shape), where len(*grid_shape) is x_dim

        F = self.conv(I_c)
        density = self.conv(M_c)
        F = F / torch.clamp(density, min=1e-8)  # normalise
        F = torch.cat([F, density], dim=0)  # shape (y_dim*2, *grid_shape)
        # F = self.resizer(F.permute(1, 2, 0)).permute(
        #     2, 0, 1
        # )  # shape (embedded_dim, *grid_shape)

        return F


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
    """Represents a ConvCNP decoder for off-the-grid data."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cnn_chans: List[int],
        embedded_dim: int,
        kernel_size: int,
        conv: nn.Module = nn.Conv1d,
        nonlinearity: nn.Module = nn.ReLU(),
        normalisation: nn.Module = nn.Identity,  # nn.BatchNorm1d
        decoder_lengthscale: float = 0.01,
        **conv_layer_kwargs,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        if cnn_chans[-1] != 2:
            cnn_chans.append(2)  # output channels for mu and sigma

        cnn_chans = [embedded_dim] + cnn_chans

        self.cnn = CNN(
            cnn_chans,
            kernel_size,
            conv,
            nonlinearity,
            normalisation,
            **conv_layer_kwargs,
        )

        self.mean_layer = SetConv(
            1, y_dim, train_lengthscale=True, lengthscale=decoder_lengthscale
        )

        self.log_sigma_layer = SetConv(
            1, y_dim, train_lengthscale=True, lengthscale=decoder_lengthscale
        )

        self.nonlinearity = nonlinearity

    def forward(
        self,
        E: Tuple[torch.Tensor, torch.Tensor],
        x_t: torch.Tensor,
    ) -> torch.distributions.Distribution:

        z_c, x_grid = E
        F = self.cnn(z_c.unsqueeze(0)).squeeze().permute(1, 0)

        assert len(F.shape) == 2
        assert F.shape[1] == 2
        assert len(x_t.shape) == 2
        assert x_t.shape[1] == self.x_dim

        mean = self.mean_layer(x_grid.unsqueeze(1), F[:, 0].unsqueeze(1), x_t)
        sigma = self.log_sigma_layer(
            x_grid.unsqueeze(1), F[:, 1].unsqueeze(1), x_t
        ).exp()

        return torch.distributions.Normal(mean, sigma)


class GridConvCNPDecoder(nn.Module):
    """Represents the decoder for a ConvCNP operating on-the-grid"""

    def __init__(
        self,
        x_dim: int,  # should be 2 for an image
        y_dim: int,  # should be 3 for a colour image (rgb)
        cnn_chans: List[int] = [128, 128, 128],
        embedded_dim: int = 128,
        cnn_kernel_size: int = 3,
        conv: nn.Module = nn.Conv2d,
        nonlinearity: nn.Module = nn.ReLU(),
        activation: nn.Module = NormalActivation(),
        res: bool = False,
        unet: bool = False,
        num_u_layers: int = 6,
        starting_chans: int = 16,
        pool: str = "max",
    ):
        super().__init__()

        if unet:
            self.cnn = Unet(
                embedded_dim,
                num_u_layers,
                starting_chans,
                cnn_kernel_size,
                conv,
                pool,
                nonlinearity,
            )
        else:

            cnn_chans = [embedded_dim] + cnn_chans

            self.cnn = CNN(
                cnn_chans,
                cnn_kernel_size,
                conv,
                nonlinearity,
                res,
            )

        if unet:
            in_dim = starting_chans
        else:
            in_dim = cnn_chans[-1]
        self.mlp = MLP(
            [in_dim] + [128] + [y_dim * 2],
            nonlinearity=nonlinearity,
        )

        self.embedded_dim = embedded_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.activation = activation

    def forward(self, E: torch.Tensor) -> torch.distributions.Distribution:
        assert E.shape[0] == self.embedded_dim
        assert len(E.shape) - 1 == self.x_dim
        # E is shape (embedded_dim, *grid_shape)
        F = self.cnn(
            E
        )  # shape (cnn_chans[-1], *grid_shape). if unet, cnn_chans[-1] == starting_chans
        F = self.mlp(F.permute(1, 2, 0)).permute(
            2, 0, 1
        )  # shape (y_dim * 2, *grid_shape)
        return self.activation(F, dim=0)


class BaseNP(nn.Module):
    """Represents a neural process base class"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        nonlinearity: nn.Module,
        embedded_dim: int = 64,
        noise: float = 1e-2,
        train_noise: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.nonlinearity = nonlinearity
        self.embedded_dim = embedded_dim

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

        elif self.conv:
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
            embedded_dim,
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
    """Represents a ConvCNP for off-the-grid data"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int = 64,
        cnn_chans: List[int] = [32, 32],
        conv: nn.Module = nn.Conv1d,
        kernel_size: int = 8,
        granularity: int = 64,  # discretized points per unit
        nonlinearity: nn.Module = nn.ReLU(),
        encoder_lengthscale: Optional[float] = None,
        decoder_lengthscale: Optional[float] = None,
        **conv_layer_kwargs,
    ):
        super().__init__(
            x_dim,
            y_dim,
            nonlinearity,
            embedded_dim,
        )

        if encoder_lengthscale is None:
            encoder_lengthscale = 10 / granularity

        if decoder_lengthscale is None:
            decoder_lengthscale = 1 / granularity

        self.encoder = ConvCNPEncoder(
            x_dim,
            y_dim,
            embedded_dim,
            granularity,
            encoder_lengthscale=encoder_lengthscale,
        )

        self.decoder = ConvCNPDecoder(
            x_dim,
            y_dim,
            cnn_chans,
            embedded_dim,
            kernel_size,
            conv=conv,
            nonlinearity=nn.ReLU(),
            normalisation=nn.Identity,
            decoder_lengthscale=decoder_lengthscale,
            **conv_layer_kwargs,
        )


class GridConvCNP(nn.Module):
    """Represents a ConvCNP operating on-the-grid"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int = 64,
        cnn_chans: List[int] = [64, 64],
        conv: nn.Module = nn.Conv2d,
        cnn_kernel_size: int = 3,
        conv_kernel_size: int = 3,
        nonlinearity: nn.Module = nn.ReLU(),
        res: bool = False,
        target_only_loss: bool = False,
        unet: bool = False,
        num_unet_layers: int = 6,
        unet_starting_chans: int = 16,
        pool: str = "max",
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.nonlinearity = nonlinearity
        self.target_only_loss = target_only_loss

        self.encoder = GridConvCNPEncoder(
            x_dim,  # should be 2 for an image
            y_dim,  # should be 3 for a colour image (rgb)
            embedded_dim,
            conv_kernel_size,
            conv=conv,
        )

        self.decoder = GridConvCNPDecoder(
            x_dim,  # should be 2 for an image
            y_dim,  # should be 3 for a colour image (rgb)
            cnn_chans=cnn_chans,
            embedded_dim=embedded_dim,
            cnn_kernel_size=cnn_kernel_size,
            conv=conv,
            nonlinearity=nonlinearity,
            activation=NormalActivation(),
            res=res,
            unet=unet,
            num_u_layers=num_unet_layers,
            starting_chans=unet_starting_chans,
            pool=pool,
        )

    def forward(
        self, I: torch.Tensor, M_c: torch.Tensor
    ) -> torch.distributions.Distribution:
        assert I.shape[-2:] == M_c.shape
        return self.decoder(self.encoder(I, M_c))

    def loss(
        self, I: torch.Tensor, M_c: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dist = self.forward(I, M_c)
        if self.target_only_loss:
            M_t = (~(M_c.bool())).double()
            ll = (M_t * dist.log_prob(I)).sum()
        else:
            ll = dist.log_prob(I).sum()
        metrics = {"ll": ll.item()}
        return (-ll / torch.numel(M_c)), metrics
