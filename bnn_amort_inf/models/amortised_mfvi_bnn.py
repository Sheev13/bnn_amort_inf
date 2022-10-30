from typing import Dict, List, Optional, Tuple

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

    # @property
    # def cache(self):
    #     return self._cache

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
        # w_dist is shape (batch_size, input_dim * output_dim)
        w_dist = self.inference_network(torch.cat((x, y), dim=-1))
        w_mus = w_dist.loc
        w_sigmas = w_dist.scale
        # product over datapoints
        w_mu, w_sigma = self._gauss_prod(w_mus, w_sigmas)
        # reshape to match prior
        w_mu = w_mu.reshape((self.output_dim, self.input_dim))
        w_sigma = w_sigma.reshape((self.output_dim, self.input_dim))
        q = torch.distributions.Normal(w_mu, w_sigma)

        assert len(q.loc.shape) == 2
        assert q.loc.shape[0] == self.output_dim
        assert q.loc.shape[1] == self.input_dim

        return q

    def kl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        qw = qw = self.qw(x, y)
        return kl_divergence(qw, self.pw).sum(-1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        F_test: torch.Tensor,
    ) -> torch.Tensor:
        """Computes and samples from q(w), carries out forward pass"""
        num_samples = F_test.shape[0]

        qw = self.qw(x, y)
        q_mus = qw.loc.unsqueeze(0).repeat(num_samples, 1, 1)
        q_sigmas = qw.scale.unsqueeze(0).repeat(num_samples, 1, 1)
        qw = torch.distributions.Normal(q_mus, q_sigmas)
        # w is shape (num_samples, output_dim, input_dim)
        w = qw.rsample()

        # self._cache["w"] = w
        # self._cache["kl"] = kl

        # (num_samples, batch_size, output_dim).
        F_test = F_test @ w.transpose(-1, -2)

        return F_test


class AmortisedMFVIBNN(nn.Module):
    """Represents the full amortised factorised-Gaussian BNN"""

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
        super().__init__()

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

        self.log_noise = nn.Parameter(
            torch.log(torch.tensor(noise)), requires_grad=train_noise
        )

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.nonlinearity = nonlinearity

    @property
    def noise(self):
        return self.log_noise.exp()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:

        assert len(x.shape) == len(y.shape) == 2
        assert x.shape[-1] == self.x_dim
        assert y.shape[-1] == self.y_dim

        assert len(x_test.shape) == 2
        assert x_test.shape[-1] == self.x_dim
        F_test = x_test.unsqueeze(0).repeat(num_samples, 1, 1)

        for i, layer in enumerate(self.layers):

            # incorporate bias term
            F_test_ones = torch.ones(F_test.shape[:-1]).unsqueeze(-1)
            F_test = torch.cat((F_test, F_test_ones), dim=-1)
            F_test = layer(x, y, F_test)

            if i != len(self.layers) - 1:
                F_test = self.nonlinearity(F_test)
            else:
                F_test = nn.Identity()(F_test)

        # probably assert some shape stuff here for good measure

        return F_test

    def exp_ll(self, F: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len(y.shape) == 2
        assert y.shape[-1] == self.y_dim
        y = y.unsqueeze(0).repeat(F.shape[0], 1, 1)
        assert y.shape == F.shape

        # (num_samples, N, output_dim)
        log_probs = torch.distributions.normal.Normal(F, self.noise).log_prob(y)
        return log_probs.sum(-1).sum(-1)

    def kl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == len(y.shape) == 2
        assert x.shape[-1] == self.x_dim
        assert y.shape[-1] == self.y_dim

        kl_total = None
        for layer in self.layers:
            kl = layer.kl(x, y)
            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        return kl_total

    def elbo(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        F = self(x, y, x_test=x, num_samples=num_samples)
        kl = self.kl(x, y)

        exp_ll = self.exp_ll(F, y).mean(0)
        kl = kl.mean(0)
        elbo = exp_ll - kl

        metrics = {
            "elbo": elbo.item(),
            "exp_ll": exp_ll.item(),
            "kl": kl.item(),
        }
        return elbo, metrics

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        elbo, metrics = self.elbo(x, y, num_samples)
        return (-elbo / x.shape[0]), metrics
