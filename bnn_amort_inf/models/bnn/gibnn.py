from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .bnn import BaseBNN
from .gibnn_layers import AmortisedGIBNNLayer, FinalGIBNNLayer, FreeFormGIBNNLayer


class GIBNN(BaseBNN):
    """Represents the standard Global Inducing Point BNN"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        num_inducing: int,
        inducing_points: Optional[torch.Tensor] = None,
        nonlinearity: nn.Module = nn.ReLU(),
        pws: Optional[List[torch.distributions.Normal]] = None,
        noise: float = 1.0,
        train_noise: bool = False,
    ):
        super().__init__(x_dim, y_dim, hidden_dims, nonlinearity, noise, train_noise)

        self.num_inducing = num_inducing

        dims = [x_dim] + hidden_dims + [y_dim]
        if pws is None:
            pws = [None] * (len(dims) - 1)
        else:
            assert len(pws) == (len(dims) - 1)

        if inducing_points is None:
            self.inducing_points = nn.Parameter(torch.randn(num_inducing, x_dim))
        else:
            assert inducing_points.shape == torch.Size((num_inducing, x_dim))
            self.inducing_points = nn.Parameter(inducing_points)

        # ModuleList for storing GIBNNLayers.
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                FreeFormGIBNNLayer(
                    dims[i] + 1,
                    dims[i + 1],
                    num_pseudo=num_inducing,
                    pw=pws[i],
                )
            )

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert len(x.shape) == 2
        assert x.shape[-1] == self.x_dim

        # (num_samples, N, input_dim).
        F = x.unsqueeze(0).repeat(num_samples, 1, 1)
        U = self.inducing_points.unsqueeze(0).repeat(num_samples, 1, 1)

        kl_total = None
        for i, layer in enumerate(self.layers):
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)
            U_ones = torch.ones(U.shape[:-1]).unsqueeze(-1)
            U = torch.cat((U, U_ones), dim=-1)

            # Sample GIBNNLayer.
            layer(U)
            qw = layer.cache["qw"]

            # (num_samples, output_dim, input_dim).
            w = qw.rsample()

            # (num_samples).
            mv_pw = torch.distributions.MultivariateNormal(
                layer.pw.loc, layer.pw.scale.pow(2).diag_embed()
            )
            kl = torch.distributions.kl.kl_divergence(qw, mv_pw).sum(-1)

            # (num_samples, batch_size, output_dim).
            F = F @ w.transpose(-1, -2)
            U = U @ w.transpose(-1, -2)
            if i < len(self.layers) - 1:
                F = self.nonlinearity(F)
                U = self.nonlinearity(U)

            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        return F, kl_total


class AmortisedGIBNN(BaseBNN):
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
        data: Optional[
            str
        ] = None,  # not None if np loss in use. Specifies if context or union dataset
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
                layer(U=F, data=data, y=y, noise=self.noise)
            else:
                layer(U=F, data=data, x=x, y=y)

            if data is None:
                qw = layer.cache["qw"]
            else:
                qw = layer.cache["qw_" + data]

            # (num_samples, output_dim, input_dim).
            w = qw.rsample()

            # (num_samples).
            mv_pw = torch.distributions.MultivariateNormal(
                layer.pw.loc, layer.pw.scale.pow(2).diag_embed()
            )
            kl = torch.distributions.kl.kl_divergence(qw, mv_pw).sum(-1)

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

    def np_loss(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(x_c.shape) == 2
        assert len(y_c.shape) == 2
        assert len(x_t.shape) == 2
        assert len(y_t.shape) == 2
        assert x_c.shape[-1] == self.x_dim
        assert y_c.shape[-1] == self.y_dim
        assert x_t.shape[-1] == self.x_dim
        assert y_t.shape[-1] == self.y_dim

        x = torch.cat((x_c, x_t), dim=0)
        y = torch.cat((y_c, y_t), dim=0)

        F_t_u = self(x, y, x_test=x_t, num_samples=num_samples, data="u")[2]
        F_t_c = self(x_c, y_c, x_test=x_t, num_samples=num_samples, data="c")[2]

        # np_kl_total = KL(q(W|D)||q(W|D_c))
        np_kl_total = None
        for layer in self.layers:
            qw_c = layer.cache["qw_c"]
            qw_u = layer.cache["qw_u"]

            np_kl = torch.distributions.kl.kl_divergence(qw_u, qw_c).sum(-1)

            if np_kl_total is None:
                np_kl_total = np_kl
            else:
                np_kl_total += np_kl

        exp_ll_t = self.exp_ll(F_t_u, y_t).mean(0)  # F_t_u: correct, F_t_c: intuitive
        np_kl_total = np_kl_total.mean(0)
        loss = exp_ll_t - np_kl_total

        metrics = {
            "loss": loss.item(),
            "exp_ll_t": exp_ll_t.item(),
            "np_kl": np_kl_total.item(),
            "noise": self.noise.item(),
        }

        return (-loss / x_t.shape[0]), metrics
