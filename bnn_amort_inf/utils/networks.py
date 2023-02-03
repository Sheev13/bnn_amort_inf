from typing import List

import torch
from torch import nn


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


class CNN(nn.Module):
    def __init__(
        self,
        dims: List[int],
        kernel_size: int,
        conv: nn.Module = nn.Conv1d,
        nonlinearity: nn.Module = nn.ReLU(),
        normalisation: nn.Module = nn.Identity,  # e.g. nn.BatchNorm1d
        **conv_layer_kwargs,
    ):
        super().__init__()

        padding = "same"

        net = []
        for i in range(len(dims) - 1):
            net.append(normalisation(dims[i]))
            net.append(
                conv(
                    dims[i],
                    dims[i + 1],
                    kernel_size,
                    padding=padding,
                    **conv_layer_kwargs,
                )
            )
            if i < len(dims) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        self.func = nn.Sequential(nn.Linear(self.in_chan, self.out_dim))
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

        F = torch.cat([density, y_c], dim=-1)  # shape (num_c, in_chan)
        F = F.unsqueeze(1) * w  # shape (num_c, num_grid, in_chan)
        F = F.sum(0)  # shape (num_grid, in_chan)

        # normalise convolution using density channel
        density, conv = F[:, : self.in_dim], F[:, self.in_dim :]
        norm_conv = conv / (density + 1e-8)
        F = torch.cat([density, norm_conv], dim=-1)

        F = self.func(F)  # shape (num_grid, out_chan)
        return F
