from abc import ABC
from typing import List, Optional

import torch
from torch import nn

from ...utils.activations import NormalActivation
from ...utils.networks import MLP
from .bnn_layers import BaseBNNLayer


class GIBNNLayer(BaseBNNLayer, ABC):
    """Represents a single layer of a GI-BNN."""

    def pseudo_likelihood(self, *args, **kwargs) -> torch.distributions.Normal:
        raise NotImplementedError

    def qw(
        self, U: torch.Tensor, pseudo_likelihood: torch.distributions.Normal
    ) -> torch.distributions.MultivariateNormal:
        assert len(U.shape) == 3
        assert U.shape[-1] == self.input_dim

        pseudo_mu = pseudo_likelihood.loc.transpose(-1, -2)  # (output_dim, N).
        pseudo_prec = pseudo_likelihood.scale.transpose(-1, -2) ** (
            -2
        )  # (output_dim, N).
        assert pseudo_mu.shape[-1] == U.shape[-2]

        # (num_samples, 1, N, input_dim).
        U_ = U.unsqueeze(1)

        #  (1, output_dim, 1, N).
        pseudo_prec_ = pseudo_prec.unsqueeze(0).unsqueeze(-2)

        # (1, output_dim, N, 1).
        pseudo_mu_ = pseudo_mu.unsqueeze(0).unsqueeze(-1)

        # (num_samples, output_dim, input_dim, batch_size)
        UTL = U_.transpose(-1, -2) * pseudo_prec_

        # FTLF is shape (num_samples, output_dim, input_dim, input_dim)
        UTLU = UTL @ U_

        # FTLv is shape (num_samples, output_dim, input_dim, 1)
        UTLv = UTL @ pseudo_mu_

        # (1, output_dim, input_dim, input_dim)
        prior_prec_ = (self.pw.scale ** (-2)).diag_embed().unsqueeze(0)

        q_prec = prior_prec_ + UTLU

        q_prec_chol = torch.linalg.cholesky(q_prec)
        q_cov = torch.cholesky_inverse(q_prec_chol)
        q_mu = (q_cov @ UTLv).squeeze(-1)
        return torch.distributions.MultivariateNormal(q_mu, q_cov)

    def forward(
        self, U: torch.Tensor, data: Optional[str] = None, *args, **kwargs
    ):  # pylint: disable=arguments-differ
        """Computes q(w) and stores in cache."""
        pseudo_likelihood = self.pseudo_likelihood(*args, **kwargs)
        qw = self.qw(U, pseudo_likelihood)
        if data is None:
            self._cache["qw"] = qw
        else:
            self._cache["qw_" + data] = qw


class FreeFormGIBNNLayer(GIBNNLayer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_pseudo: int,
        pseudo_mu: Optional[torch.Tensor] = None,
        pseudo_prec: Optional[torch.Tensor] = None,
        pw: Optional[torch.distributions.Normal] = None,
    ):
        super().__init__(input_dim, output_dim, pw)

        if pseudo_mu is None:
            self.pseudo_mu = nn.Parameter(torch.randn(num_pseudo, output_dim))
        else:
            assert pseudo_mu.shape[1] == output_dim
            assert pseudo_mu.shape[0] == num_pseudo
            self.pseudo_mu = nn.Parameter(pseudo_mu)

        if pseudo_prec is None:
            self.pseudo_logprec = nn.Parameter(
                (torch.ones(num_pseudo, output_dim) * 1).log()
            )
        else:
            assert pseudo_prec.shape[1] == output_dim
            assert pseudo_prec.shape[0] == num_pseudo
            self.pseudo_logprec = nn.Parameter(pseudo_prec.log())

    def pseudo_likelihood(  # pylint: disable=arguments-differ
        self,
    ) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.pseudo_mu, (-self.pseudo_logprec).exp())


class FinalGIBNNLayer(GIBNNLayer):
    def pseudo_likelihood(  # pylint: disable=arguments-differ
        self, y: torch.Tensor, noise: torch.Tensor
    ) -> torch.distributions.Normal:
        return torch.distributions.Normal(y, noise)


class AmortisedGIBNNLayer(GIBNNLayer):
    """Represents a single layer of a Bayesian neural network with amortisation."""

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
            [x_dim + y_dim] + in_hidden_dims + [2 * output_dim],
            nonlinearity=in_nonlinearity,
            activation=NormalActivation(),
        )

    def pseudo_likelihood(  # pylint: disable=arguments-differ
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.distributions.Normal:
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[-1] == self.x_dim
        assert y.shape[-1] == self.y_dim

        return self.inference_network(torch.cat((x, y), dim=-1))
