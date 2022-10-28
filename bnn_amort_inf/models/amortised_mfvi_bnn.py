from typing import Dict, List, Optional

import torch
from torch import nn
from torch.distributions.kl import kl_divergence

from ..utils.activations import NormalActivation
from ..utils.networks import MLP


class AmortisedMFVIBNNLayer(nn.Module):
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pw is None:
            self.pw = torch.distributions.Normal(
                torch.zeros(output_dim, input_dim), torch.ones(output_dim, input_dim)
            )
        else:
            assert isinstance(pw, torch.distributions.Normal)
            assert pw.loc.shape == torch.Size((output_dim, input_dim))
            self.pw = pw

        # self._cache: Dict = {}

        self.inference_network = MLP(
            [x_dim + y_dim] + in_hidden_dims + [2 * input_dim * output_dim],
            nonlinearity=in_nonlinearity,
            activation=NormalActivation(),
        )

    @property
    def cache(self):
        return self._cache

    def _gauss_prod(
        mus: torch.Tensor, sigmas: torch.Tensor
    ):  # what is the best practice for where to implement this?
        assert len(mus.shape) == 2
        assert mus.shape == sigmas.shape

        np1 = (mus / (sigmas**2)).sum(0)
        np2 = (-1 / (2 * sigmas**2)).sum(0)

        mu = -np1 / (2 * np2)
        sigma = (-1 / (2 * np2)) ** 0.5

        return mu, sigma

    def qw(self, x: torch.Tensor, y: torch.Tensor, num_samples: torch.Tensor):
        # w_dist is shape (batch_size, input_dim * output_dim)
        w_dist = self.inference_network(torch.cat((x, y), dim=-1))
        w_mus = w_dist.loc
        w_sigmas = w_dist.scale
        # product over datapoints
        w_mu, w_sigma = self._gauss_prod(w_mus, w_sigmas)
        q = torch.distributions.Normal(w_mu, w_sigma)
        # reshape to match prior
        q = q.reshape((self.output_dim, self.input_dim))

        assert len(q.shape) == 2
        assert q.shape[0] == self.output_dim
        assert q.shape[1] == self.input_dim

        return q

    def kl(self, x: torch.Tensor, y: torch.Tensor):
        qw = qw = self.qw(x, y)
        return kl_divergence(qw, self.pw)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor):
        """Computes and samples from q(w), carries out forward pass"""
        num_samples = x_test.shape[0]

        qw = self.qw(x, y)
        qw = qw.unsqueeze(0).repeat(num_samples, 1, 1)
        # w is shape (num_samples, output_dim, input_dim)
        w = qw.rsample()

        # self._cache["w"] = w
        # self._cache["kl"] = kl

        # (num_samples, batch_size, output_dim).
        x_test = x_test @ w.transpose(-1, -2)

        return x_test

    class AmortisedMFVIBNN(nn.Module):
        """Represents the full amortised factorised-Gaussian BNN"""

        def __init__(
            self,
        ):
            super().__init__()
