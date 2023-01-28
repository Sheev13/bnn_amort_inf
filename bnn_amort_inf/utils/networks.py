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


class ConvLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        conv: nn.Module,
        nonlinearity: nn.Module,
        normalisation: nn.Module,
        **kwargs,
    ):
        super().__init__()

        self.nonlinearity = nonlinearity
        padding = kernel_size // 2
        self.conv = conv(input_dim, output_dim, kernel_size, padding=padding, **kwargs)
        self.normalisation = normalisation(input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(self.nonlinearity(self.normalisation(x)))


class CNN(nn.Module):
    def __init__(
        self,
        dims: List[int],
        kernel_size: int,
        conv: nn.Module = nn.Conv1d,
        nonlinearity: nn.Module = nn.ReLU(),
        normalisation: nn.Module = nn.Identity,  # nn.BatchNorm1d
        **conv_layer_kwargs,
    ):
        super().__init__()

        net = []
        for i in range(len(dims) - 1):
            net.append(
                ConvLayer(
                    dims[i],
                    dims[i + 1],
                    kernel_size,
                    conv,
                    nonlinearity,
                    normalisation,
                    **conv_layer_kwargs,
                )
            )

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetConv(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self):
        pass
