from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .gibnn import BaseGIBNN
from .gibnn_layers import AmortisedGIBNNLayer, FinalGIBNNLayer


class AmortisedGIBNN(BaseGIBNN):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        in_hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        pws: Optional[List[torch.distributions.Normal]] = None,
        in_nonlinearity: nn.Module = nn.ReLU(),
        noise: float = 1.0,
        train_noise: bool = False,
    ):
        amortised = True
        super().__init__(
            x_dim, y_dim, hidden_dims, nonlinearity, noise, train_noise, amortised
        )

        dims = [x_dim] + hidden_dims + [y_dim]
        if pws is None:
            pws = [None] * (len(dims) - 1)
        else:
            assert len(pws) == (len(dims) - 1)

        # ModuleList for storing GIBNNLayers.
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.layers.append(
                AmortisedGIBNNLayer(
                    x_dim,
                    y_dim,
                    dims[i] + 1,
                    dims[i + 1],
                    in_hidden_dims=in_hidden_dims,
                    in_nonlinearity=in_nonlinearity,
                    pw=pws[i],
                )
            )
        self.layers.append(
            FinalGIBNNLayer(
                dims[-2] + 1,
                dims[-1],
                pw=pws[-1],
            )
        )

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape[-1] == self.x_dim
        assert y.shape[-1] == self.y_dim

        # (num_samples, N, input_dim).
        F = x.unsqueeze(0).repeat(num_samples, 1, 1)

        if x_test is not None:
            assert len(x_test.shape) == 2
            assert x_test.shape[-1] == self.x_dim
            F_test = x_test.unsqueeze(0).repeat(num_samples, 1, 1)
        else:
            F_test = None

        kl_total = None
        for i, layer in enumerate(self.layers):
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)

            if F_test is not None:
                F_test_ones = torch.ones(F_test.shape[:-1]).unsqueeze(-1)
                F_test = torch.cat((F_test, F_test_ones), dim=-1)

            # Sample GIBNNLayer.
            if i == (len(self.layers) - 1):
                layer(U=F, y=y, noise=self.noise)
            else:
                layer(U=F, x=x, y=y)

            w = layer.cache["w"]
            kl = layer.cache["kl"]

            # (num_samples, batch_size, output_dim).
            F = F @ w.transpose(-1, -2)
            if i < len(self.layers) - 1:
                F = self.nonlinearity(F)

            if F_test is not None:
                F_test = F_test @ w.transpose(-1, -2)
                if i < len(self.layers) - 1:
                    F_test = self.nonlinearity(F_test)

            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        return F, kl_total, F_test
