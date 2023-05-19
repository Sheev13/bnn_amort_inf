from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        activation: nn.Module = nn.Identity(),
        bias: bool = True,
    ):
        super().__init__()

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1], bias))

            if i == len(dims) - 2:
                net.append(activation)
            else:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        conv: nn.Module,
        kernel_size: int,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.nonlinearity = nonlinearity

        self.conv = conv(in_chan, out_chan, kernel_size, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.nonlinearity(x))


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        conv: nn.Module,
        kernel_size: int,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.nonlinearity = nonlinearity

        self.conv_depthwise = conv(
            in_chan, in_chan, kernel_size, padding="same", groups=in_chan
        )

        self.conv_pointwise = conv(in_chan, out_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F = self.conv_depthwise(self.nonlinearity(x))
        F = F + x
        return self.conv_pointwise(
            self.nonlinearity(F)
        )  # might need F.contiguous() here


class CNN(nn.Module):
    def __init__(
        self,
        chans: List[int],
        kernel_size: int,
        conv: nn.Module = nn.Conv1d,
        nonlinearity: nn.Module = nn.ReLU(),
        res: bool = False,
    ):
        super().__init__()

        if res:
            conv_block = ResConvBlock
        else:
            conv_block = ConvBlock

        net = []
        for i in range(len(chans) - 1):
            # if chans[i] != chans[i + 1]:
            #     conv_block = (
            #         ConvBlock
            #     )
            net.append(
                conv_block(
                    chans[i],
                    chans[i + 1],
                    conv,
                    kernel_size,
                    nonlinearity,
                )
            )
            # if res:
            #     conv_block = ResConvBlock

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_chans: int,  # channels for first block
        num_u_layers: int,
        starting_chans: int,  # number of channels in first layer
        kernel_size: int,
        conv: nn.Module = nn.Conv2d,
        pool: str = "max",
        nonlinearity: nn.Module = nn.ReLU(),
        out_chans: Optional[int] = None,
    ):
        super().__init__()

        assert pool in ["max", "avg"]
        dim = [nn.Conv1d, nn.Conv2d, nn.Conv3d].index(conv) + 1
        pools = {
            "max": [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d],
            "avg": [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d],
        }
        self.pool = pools[pool][dim - 1](2)  # use pooling size of 2

        upsamp_modes = ["linear", "bilinear", "trilinear"]
        self.upsamp_mode = upsamp_modes[dim - 1]

        assert num_u_layers % 2 == 0
        self.num_u_blocks = num_u_layers - 1
        chans = [starting_chans * 2**i for i in range(num_u_layers // 2)]
        self.u_chans = chans + chans[::-1]

        self.in_block = ConvBlock(
            in_chans,
            starting_chans,
            conv,
            kernel_size,
            nonlinearity,
        )

        self.out_block = nn.Identity()
        if out_chans is not None:
            self.out_block = ConvBlock(
                starting_chans, out_chans, conv, kernel_size, nonlinearity
            )

        self.u_blocks = []
        for i in range(self.num_u_blocks):
            double = int(i > self.num_u_blocks // 2)
            self.u_blocks.append(
                ConvBlock(
                    self.u_chans[i] * 2**double,
                    self.u_chans[i + 1],
                    conv,
                    kernel_size,
                    nonlinearity,
                )
            )

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        half_u_blocks = self.num_u_blocks // 2
        residuals = [None] * (half_u_blocks)
        # Down
        for i in range(half_u_blocks):
            x = self.u_blocks[i](x)
            residuals[i] = x
            x = self.pool(x)

        # Bottleneck
        x = self.u_blocks[half_u_blocks](x)

        # Up
        for i in range((half_u_blocks) + 1, self.num_u_blocks):
            if self.dim > 1:
                x = x.unsqueeze(0)
            x = F.interpolate(
                x,
                mode=self.upsamp_mode,
                align_corners=True,
                size=(residuals[half_u_blocks - i].shape[-self.dim :]),
            )
            if self.dim > 1:
                x = x.squeeze(0)
                dim = 0
            else:
                dim = 1
            x = torch.cat(
                (x, residuals[half_u_blocks - i]), dim=dim
            )  # concat on channels
            x = self.u_blocks[i](x)

        return self.out_block(x)


class SetConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        train_lengthscale: bool = True,
        lengthscale: float = 5e-1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_chan = in_dim + out_dim
        self.resizer = nn.Sequential(nn.Linear(self.in_chan, self.out_dim))
        self.log_sigma = nn.Parameter(
            torch.tensor(lengthscale).log() * torch.ones(self.in_chan),
            requires_grad=train_lengthscale,
        )

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def rbf(self, dists):
        return (-0.5 * dists.unsqueeze(-1) / self.sigma**2).exp()

    def forward(self, x_c, y_c, x_grid):
        assert x_c.shape[1] == self.in_dim
        if len(x_grid.shape) == 1:
            x_grid = x_grid.unsqueeze(1)

        dists = torch.cdist(x_c, x_grid, p=2).squeeze(1)  # shape (num_c, num_grid)
        w = self.rbf(dists)  # shape (num_c, num_grid, in_chan)
        density = torch.ones_like(x_c)

        F = torch.cat(
            [density, y_c.repeat(1, self.out_dim)], dim=-1
        )  # shape (num_c, in_chan)     repeat here is new
        F = F.unsqueeze(1) * w  # shape (num_c, num_grid, in_chan)
        F = F.sum(0)  # shape (num_grid, in_chan)

        # normalise convolution using density channel
        density, conv = F[..., : self.in_dim], F[..., self.in_dim :]
        norm_conv = conv / (density + 1e-8)
        F = torch.cat([density, norm_conv], dim=-1)

        F = self.resizer(F)  # shape (num_grid, out_dim)
        return F
