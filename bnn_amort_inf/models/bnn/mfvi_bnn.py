from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...utils.activations import NormalActivation
from ...utils.networks import MLP
from .bnn import BaseBNN
from .bnn_layers import BaseBNNLayer


class AmortisedMFVIBNNLayer(BaseBNNLayer):
    """Represents a layer of an amortised factorised-Gaussian BNN"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        input_dim: int,
        output_dim: int,
        in_hidden_dims: List[int],
        in_nonlinearity: nn.Module = nn.ReLU(),
        pw: Optional[torch.distributions.Normal] = None,
    ):
        super().__init__(input_dim, output_dim, pw)

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.inference_network = MLP(
            [x_dim + y_dim] + in_hidden_dims + [2 * input_dim * output_dim],
            nonlinearity=in_nonlinearity,
            activation=NormalActivation(),
        )

    def _gauss_prod(  # what is the best practice for where to implement this?
        self, mus: torch.Tensor, sigmas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(mus.shape) == 2
        assert mus.shape == sigmas.shape

        np1 = (mus / (sigmas**2)).sum(0)
        np2 = (-1 / (2 * sigmas**2)).sum(0)

        mu = -np1 / (2 * np2)
        sigma = (-1 / (2 * np2)) ** 0.5

        return mu, sigma

    def qw(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # dists is shape (batch_size, input_dim * output_dim).
        dists = self.inference_network(torch.cat((x, y), dim=-1))

        lw_np1 = (
            (dists.loc / (dists.scale**2))
            .sum(0)
            .reshape((self.output_dim, self.input_dim))
        )
        lw_np2 = (
            (-1 / (2 * dists.scale**2))
            .sum(0)
            .reshape((self.output_dim, self.input_dim))
        )

        pw_np1 = self.pw.loc / (self.pw.scale**2)
        pw_np2 = -1 / (2 * self.pw.scale**2)

        qw_np1 = pw_np1 + lw_np1
        qw_np2 = pw_np2 + lw_np2
        qw_loc = -qw_np1 / (2 * qw_np2)
        qw_scale = (-1 / (2 * qw_np2)) ** 0.5
        return torch.distributions.Normal(qw_loc, qw_scale)

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """Computes q(w) and stores in cache."""
        self._cache["qw"] = self.qw(x, y)


class AmortisedMFVIBNN(BaseBNN):
    """Represents the full amortised factorised-Gaussian BNN"""

    amortised = True

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
        super().__init__(x_dim, y_dim, hidden_dims, nonlinearity, noise, train_noise)

        dims = [x_dim] + hidden_dims + [y_dim]
        if pws is None:
            pws = [None] * (len(dims) - 1)
        else:
            assert len(pws) == (len(dims) - 1)

        self.layers = nn.ModuleList()

        for i in range(len(dims) - 2):
            self.layers.append(
                AmortisedMFVIBNNLayer(
                    x_dim,
                    y_dim,
                    dims[i] + 1,  # + 1 for bias
                    dims[i + 1],
                    in_hidden_dims=in_hidden_dims,
                    in_nonlinearity=in_nonlinearity,
                    pw=pws[i],
                )
            )

        self.layers.append(
            AmortisedMFVIBNNLayer(
                x_dim,
                y_dim,
                dims[-2] + 1,
                dims[-1],
                in_hidden_dims=in_hidden_dims,
                in_nonlinearity=in_nonlinearity,
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

        assert len(x.shape) == len(y.shape) == 2
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

            # Sample BNNLayer.
            if i == (len(self.layers) - 1):
                layer(x, y)
            else:
                layer(x, y)

            qw = layer.cache["qw"]

            # (num_samples, output_dim, input_dim).
            w = qw.rsample((num_samples,))
            kl = torch.distributions.kl.kl_divergence(qw, layer.pw).sum()

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
