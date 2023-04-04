from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ...utils.networks import CNN, MLP, SetConv, Unet
from ..likelihoods.base import Likelihood
from ..likelihoods.normal import NormalLikelihood
from .base import BaseNP


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, x):
            return nn.functional.conv2d(
                x,
                self.weight.exp(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


class ConvCNPEncoder(nn.Module):
    """Represents a ConvCNP encoder for off-the-grid data.."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int,
        granularity: int,
        lengthscale: float = 0.1,
    ):
        super().__init__()

        self.set_conv = SetConv(
            x_dim,
            embedded_dim,
            train_lengthscale=True,
            lengthscale=lengthscale,
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

        F_c = self.set_conv(x_c, y_c, x_grid).permute(1, 0)

        assert F_c.shape[0] == self.embedded_dim
        assert F_c.shape[1] == num_points

        return (F_c, x_grid)


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

    def forward(self, x: torch.Tensor, mask_c: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == self.x_dim + 1  # extra dimension for colour channels
        assert x.shape[0] == self.y_dim
        if len(mask_c.shape) == self.x_dim:
            mask_c = mask_c.unsqueeze(0).repeat(x.shape[0], 1, 1)
        assert len(mask_c.shape) == len(x.shape)

        x_c = x * mask_c  # get context pixels from image and context mask
        # shape (y_dim, *grid_shape), where len(*grid_shape) is x_dim

        z_c = self.conv(x_c)
        density_c = self.conv(mask_c)
        z_c = z_c / torch.clamp(density_c, min=1e-8)  # normalise
        z_c = torch.cat([z_c, density_c], dim=0)  # shape (y_dim*2, *grid_shape)
        z_c = self.resizer(z_c.permute(1, 2, 0)).permute(
            2, 0, 1
        )  # shape (embedded_dim, *grid_shape)

        return z_c


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
        lengthscale: float = 0.01,
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
        **conv_layer_kwargs,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim
        self.likelihood = likelihood

        # if cnn_chans[-1] != 2:
        #     cnn_chans.append(2)  # output channels for mu and sigma
        if cnn_chans[-1] != self.likelihood.out_dim_multiplier:
            cnn_chans.append(
                self.likelihood.out_dim_multiplier
            )  # Single output channel.

        cnn_chans = [embedded_dim] + cnn_chans

        self.cnn = CNN(
            cnn_chans,
            kernel_size,
            conv,
            nonlinearity,
            normalisation,
            **conv_layer_kwargs,
        )
        self.set_convs = [
            SetConv(1, y_dim, train_lengthscale=True, lengthscale=lengthscale)
            for _ in range(self.likelihood.out_dim_multiplier)
        ]

        # self.mean_layer = SetConv(
        #     1, y_dim, train_lengthscale=True, lengthscale=decoder_lengthscale
        # )

        # self.log_sigma_layer = SetConv(
        #     1, y_dim, train_lengthscale=True, lengthscale=decoder_lengthscale
        # )

    def forward(
        self,
        z_c: torch.Tensor,
        x_grid: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.distributions.Distribution:
        z_c_proc = self.cnn(z_c.unsqueeze(0)).squeeze(0).permute(1, 0)

        assert len(z_c_proc.shape) == 2
        assert z_c_proc.shape[1] == self.likelihood.out_dim_multiplier
        assert len(x_t.shape) == 2
        assert x_t.shape[1] == self.x_dim

        return self.likelihood(
            torch.cat(
                [
                    self.set_convs[i](
                        x_grid.unsqueeze(1), z_c_proc[:, i].unsqueeze(-1), x_t
                    )
                    for i in range(self.likelihood.out_dim_multiplier)
                ],
                dim=-1,
            )
        )

        # mean = self.mean_layer(x_grid.unsqueeze(1), z_c[:, 0].unsqueeze(1), x_t)
        # sigma = self.log_sigma_layer(
        #     x_grid.unsqueeze(1), z_c[:, 1].unsqueeze(1), x_t
        # ).exp()


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
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
        res: bool = False,
        unet: bool = False,
        num_u_layers: int = 6,
        starting_chans: int = 16,
        pool: str = "max",
    ):
        super().__init__()

        self.embedded_dim = embedded_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.likelihood = likelihood

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
            [in_dim] + [128] + [self.likelihood.out_dim_multiplier * y_dim],
            nonlinearity=nonlinearity,
        )

    def forward(self, z_c: torch.Tensor) -> torch.distributions.Distribution:
        assert z_c.shape[0] == self.embedded_dim
        # TODO: this doesn't seem correct?
        # assert len(z_c.shape) - 1 == self.x_dim
        # E is shape (embedded_dim, *grid_shape)
        z_c = self.cnn(
            z_c
        )  # shape (cnn_chans[-1], *grid_shape). if unet, cnn_chans[-1] == starting_chans
        z_c = self.mlp(z_c.permute(1, 2, 0)).permute(
            2, 0, 1
        )  # shape (y_dim * 2, *grid_shape)
        return self.likelihood(z_c)


class ConvCNP(BaseNP):
    """Represents a ConvCNP for off-the-grid data"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int = 64,
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
        cnn_chans: List[int] = [32, 32],
        conv: nn.Module = nn.Conv1d,
        kernel_size: int = 8,
        granularity: int = 64,  # discretized points per unit
        encoder_lengthscale: Optional[float] = None,
        decoder_lengthscale: Optional[float] = None,
        **conv_layer_kwargs,
    ):
        if encoder_lengthscale is None:
            encoder_lengthscale = 10 / granularity

        if decoder_lengthscale is None:
            decoder_lengthscale = 1 / granularity

        encoder = ConvCNPEncoder(
            x_dim,
            y_dim,
            embedded_dim,
            granularity,
            lengthscale=encoder_lengthscale,
        )

        decoder = ConvCNPDecoder(
            x_dim,
            y_dim,
            cnn_chans,
            embedded_dim,
            kernel_size,
            conv=conv,
            nonlinearity=nn.ReLU(),
            normalisation=nn.Identity,
            lengthscale=decoder_lengthscale,
            likelihood=likelihood,
            **conv_layer_kwargs,
        )

        super().__init__(x_dim, y_dim, embedded_dim, encoder, decoder)

    def forward(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.decoder(*self.encoder(x_c, y_c, x_t), x_t)


class GridConvCNP(BaseNP):
    """Represents a ConvCNP operating on-the-grid"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int = 64,
        likelihood: Likelihood = NormalLikelihood(noise=0.1),
        cnn_chans: List[int] = [64, 64],
        conv: nn.Module = nn.Conv2d,
        cnn_kernel_size: int = 3,
        conv_kernel_size: int = 3,
        res: bool = False,
        unet: bool = False,
        num_unet_layers: int = 6,
        unet_starting_chans: int = 16,
        pool: str = "max",
    ):
        encoder = GridConvCNPEncoder(
            x_dim,  # should be 2 for an image
            y_dim,  # should be 3 for a colour image (rgb)
            embedded_dim,
            conv_kernel_size,
            conv=conv,
        )

        decoder = GridConvCNPDecoder(
            x_dim,  # should be 2 for an image
            y_dim,  # should be 3 for a colour image (rgb)
            cnn_chans=cnn_chans,
            embedded_dim=embedded_dim,
            cnn_kernel_size=cnn_kernel_size,
            conv=conv,
            nonlinearity=nn.ReLU(),
            likelihood=likelihood,
            res=res,
            unet=unet,
            num_u_layers=num_unet_layers,
            starting_chans=unet_starting_chans,
            pool=pool,
        )

        super().__init__(x_dim, y_dim, embedded_dim, encoder, decoder)

    def forward(
        self,
        x: torch.Tensor,
        mask_c: torch.Tensor,
    ) -> torch.distributions.Distribution:
        assert x.shape[-2:] == mask_c.shape
        return self.decoder(self.encoder(x, mask_c))

    def npml_loss(
        self,
        x: torch.Tensor,
        mask_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dist = self.forward(x, mask_c)
        ll = dist.log_prob(x).sum()
        metrics = {"ll": ll.item()}
        return (-ll / torch.count_nonzero(mask_c.float())), metrics
