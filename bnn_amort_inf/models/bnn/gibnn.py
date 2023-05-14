from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ..likelihoods.base import Likelihood
from ..likelihoods.normal import NormalLikelihood
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
        final_layer_mu: Optional[torch.Tensor] = None,
        final_layer_prec: float = None,
        nonlinearity: nn.Module = nn.ReLU(),
        pws: Optional[List[torch.distributions.Normal]] = None,
        likelihood: Likelihood = NormalLikelihood(noise=1.0),
        learn_inducing_points: bool = True,
        learn_final_layer_mu: bool = True,
        learn_final_layer_prec: bool = True,
    ):
        super().__init__(x_dim, y_dim, hidden_dims, nonlinearity, likelihood)

        self.num_inducing = num_inducing

        dims = [x_dim] + hidden_dims + [y_dim]
        if pws is None:
            pws = [None] * (len(dims) - 1)
        else:
            assert len(pws) == (len(dims) - 1)

        if inducing_points is None:
            self.inducing_points = nn.Parameter(
                torch.randn(num_inducing, x_dim), requires_grad=learn_inducing_points
            )
        else:
            assert inducing_points.shape == torch.Size((num_inducing, x_dim))
            self.inducing_points = nn.Parameter(
                inducing_points, requires_grad=learn_inducing_points
            )

        # ModuleList for storing GIBNNLayers.
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            # Initialise the pseudo-precisions to be small for the final layer.
            if i == len(dims) - 2:
                if final_layer_prec is not None:
                    init_prec = final_layer_prec
                else:
                    init_prec = 1e-2
                if final_layer_mu is not None:
                    init_mu = final_layer_mu
                else:
                    init_mu = None

                if learn_final_layer_mu:
                    learn_mu = True
                else:
                    learn_mu = False

                if learn_final_layer_prec:
                    learn_prec = True
                else:
                    learn_prec = False
            else:
                init_prec = 1e-2
                init_mu = None

                learn_mu = True
                learn_prec = True

            self.layers.append(
                FreeFormGIBNNLayer(
                    dims[i] + 1,
                    dims[i + 1],
                    num_pseudo=num_inducing,
                    pw=pws[i],
                    pseudo_mu=init_mu,
                    pseudo_prec=torch.ones(self.num_inducing, dims[i + 1]) * init_prec,
                    learn_mu=learn_mu,
                    learn_prec=learn_prec,
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
        likelihood: Likelihood = NormalLikelihood(noise=1.0),
    ):
        super().__init__(x_dim, y_dim, hidden_dims, nonlinearity, likelihood)

        dims = [x_dim] + hidden_dims + [y_dim * self.likelihood.out_dim_multiplier]
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

        if hasattr(likelihood, "noise"):
            self.layers.append(
                FinalGIBNNLayer(
                    dims[-2] + 1,
                    dims[-1],
                    pw=pws[-1],
                )
            )
        else:
            self.layers.append(
                AmortisedGIBNNLayer(
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
        compute_kl: bool = True,
        cache_name: str = "qw",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
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

            # Sample GIBNNLayer.
            if i == (len(self.layers) - 1) and hasattr(self.likelihood, "noise"):
                layer(U=F, cache_name=cache_name, y=y, noise=self.likelihood.noise)
            else:
                layer(U=F, cache_name=cache_name, x=x, y=y)

            qw = layer.cache[cache_name]

            # (num_samples, output_dim, input_dim).
            w = qw.rsample()

            if compute_kl:
                # (num_samples).
                mv_pw = torch.distributions.MultivariateNormal(
                    layer.pw.loc, layer.pw.scale.pow(2).diag_embed()
                )
                kl = torch.distributions.kl.kl_divergence(qw, mv_pw).sum(-1)

                if kl_total is None:
                    kl_total = kl
                else:
                    kl_total += kl

            # (num_samples, batch_size, output_dim).
            F = F @ w.transpose(-1, -2)
            if i < len(self.layers) - 1:
                F = self.nonlinearity(F)

            if F_test is not None:
                F_test_ones = torch.ones(F_test.shape[:-1]).unsqueeze(-1)
                F_test = torch.cat((F_test, F_test_ones), dim=-1)
                F_test = F_test @ w.transpose(-1, -2)
                if i < len(self.layers) - 1:
                    F_test = self.nonlinearity(F_test)

        return F, kl_total, F_test

    def npml_loss(
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

        F_t = self(x_c, y_c, x_test=x_t, num_samples=num_samples, compute_kl=False)[2]
        qy_t = self.likelihood(F_t)
        exp_ll = (
            qy_t.log_prob(y_t.unsqueeze(0).repeat(num_samples, 1, 1)).sum()
            / num_samples
        )

        metrics = {
            "exp_ll": exp_ll.item(),
        }

        return (-exp_ll / x_t.shape[0]), metrics

    def npvi_loss(
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

        x = torch.cat((x_c, x_t), dim=-2)
        y = torch.cat((y_c, y_t), dim=-2)

        F_t = self(
            x,
            y,
            x_test=x_t,
            num_samples=num_samples,
            compute_kl=False,
            cache_name="qw_u",
        )[2]
        _ = self(x_c, y_c, num_samples=num_samples, compute_kl=False, cache_name="qw_c")

        kl = torch.as_tensor(0.0)
        for layer in self.layers:
            qw_c = layer.cache["qw_c"]
            qw_u = layer.cache["qw_u"]
            kl += torch.distributions.kl.kl_divergence(qw_u, qw_c).sum() / num_samples

        # exp_ll_t = self.exp_ll(F_t_c, y_t).mean(0)  # F_t_u: correct, F_t_c: intuitive
        qy_t = self.likelihood(F_t)
        exp_ll = (
            qy_t.log_prob(y_t.unsqueeze(0).repeat(num_samples, 1, 1)).sum()
            / num_samples
        )
        elbo = exp_ll - kl

        metrics = {
            "elbo": elbo.item(),
            "exp_ll": exp_ll.item(),
            "kl": kl.item(),
        }

        return (-elbo / x_t.shape[0]), metrics
