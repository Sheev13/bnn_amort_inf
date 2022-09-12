import torch
from torch import nn


class GILayer(nn.Module):
    """Represents a single layer of a Bayesian neural network with global inducing points"""

    def __init__(self, input_dim, output_dim, num_induce, nonlinearity):
        super(GILayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_induce = num_induce
        self.nonlinearity = nonlinearity

        # priors
        self.mu_p = torch.zeros(output_dim, input_dim)
        self.logvar_p = torch.zeros(output_dim, input_dim)
        # TODO: fix this/make it diagonal
        # TODO: make this a multivariate normal with diagonal covariance so that kl works later on
        self.prior = torch.distributions.Normal(self.mu_p, (0.5 * self.logvar_p).exp())
        self.full_prior = torch.distributions.MultivariateNormal(
            self.mu_p, (0.5 * self.logvar_p.diag_embed()).exp()
        )

        # pseudos
        self.pseud_mu = nn.Parameter(torch.zeros(output_dim, num_induce))
        self.pseud_logprec = nn.Parameter(torch.zeros(output_dim, num_induce))

    def q(self, U: torch.Tensor) -> torch.distribution.MultivariateNormal:
        # U is shape (num_samples, num_induces, input_dim).
        assert len(U.shape) == 3
        assert U.shape[1] == self.num_induce
        assert U.shape[2] == self.input_dim

        U = self.nonlinearity(U)

        # U_ is shape (num_samples, 1, num_induce, input_dim).
        U_ = U.unsqueeze(1)

        # pseud_logprec_ is shape (1, output_dim, num_induce, num_induce).
        pseud_logprec_ = self.pseud_logprec.diag_embed().unsqueeze(0)

        # pseud_mu_ is shape (1, output_dim, num_induce, 1).
        pseud_mu_ = self.pseud_mu.unsqueeze(0).unsqueeze(-1)

        # UTL is shape (num_samples, output_dim, input_dim, num_induces)
        UTL = U.transpose(-1, -2) @ pseud_logprec_.exp()

        # UTLU is shape (num_samples, output_dim, input_dim, input_dim)
        UTLU = UTL @ U_

        # UTLv is shape (num_samples, output_dim, input_dim, 1)
        UTLv = UTL @ pseud_mu_

        # prior_prec_ is shape (1, 1, input_dim, input_dim).
        prior_prec_ = (-self.logvar_p).exp().diag_embed().unsqueeze(0)

        q_prec = prior_prec_ + UTLU
        q_cov = torch.cholesky_inverse(q_prec)
        q_mu = (q_cov @ UTLv).squeeze(-1)
        return torch.distributions.MultivariateNormal(q_mu, q_cov)

    def forward(self, F, U):
        assert len(U.shape) == 3
        assert len(F.shape) == 3
        assert U.shape[1] == self.num_induce
        assert U.shape[2] == self.input_dim
        assert F.shape[2] == self.input_dim

        # # augment inputs with ones to absorb bias
        # F_ones = torch.ones(shape=(F.shape[0], 1))
        # # TODO: check this is right way around, and check dim
        # F = torch.cat([F, F_ones], dim=-1)

        # U_ones = torch.ones(shape=(U.shape[0], 1))
        # # TODO: check this is right way around, and check dim
        # U = torch.cat([U, U_ones], dim=-1)

        q = self.q(U)

        # w is shape (num_samples, output_dim, input_dim).
        w = q.rsample()

        # w_ is shape (num_samples, 1, output_dim, input_dim).
        w_ = w.unsqueeze(1)

        # kl_contribution is shape (num_samples).
        kl_contribution = torch.distributions.kl.kl_divergence(q, self.full_prior).sum(
            -1
        )

        # TODO: check that these are the right way around
        # F and U are shape (num_samples, batch_size, output_dim).
        F = (w_ @ self.nonlinearity(F).unsqueeze(-1)).squeeze(-1)
        U = (w_ @ self.nonlinearity(U).unsqueeze(-1)).squeeze(-1)

        return F, U, kl_contribution


class GINetwork(nn.Module):
    """Represents the full Global Inducing Point BNN"""

    def __init__(
        self, input_dim, hidden_dims, output_dim, inducing_points, nonlinearity=nn.ELU()
    ):
        super(GINetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        # TODO: can we initialise the inducing points internally here instead?
        self.inducing_points = nn.Parameter(inducing_points)
        self.num_induce = inducing_points.shape[0]
        self.nonlinearity = nonlinearity
        self.log_noise = nn.Parameter(torch.tensor(-1))

        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(
                    GILayer(
                        self.input_dim + 1,
                        self.hidden_dims[i],
                        num_induce=self.num_induce,
                        nonlinearity=self.nonlinearity,
                    )
                )
            elif i == len(hidden_dims):
                self.network.append(
                    GILayer(
                        self.hidden_dims[i - 1] + 1,
                        self.output_dim,
                        num_induce=self.num_induce,
                        nonlinearity=self.nonlinearity,
                    )
                )
            else:
                self.network.append(
                    GILayer(
                        self.hidden_dims[i - 1] + 1,
                        self.hidden_dims[i],
                        num_induce=self.num_induce,
                        nonlinearity=self.nonlinearity,
                    )
                )

    @property
    def noise(self):
        return torch.exp(self.log_noise)

    def forward(self, F, num_samples=1):
        assert len(F.shape) == 2
        assert F.shape[1] == self.input_dim

        # (num_samples, batch_size, input_dim).
        F = F.unsqueeze(0).repeat(num_samples, 1, 1)
        U = self.inducing_points.unsqueeze(0).repeat(num_samples, 1, 1)

        kl_total = torch.tensor(0)
        U = self.inducing_points
        for i, layer in enumerate(self.network):
            F_ones = torch.ones((F.shape[:-1], 1))
            F = torch.cat([F, F_ones], dim=-1)
            U_ones = torch.ones((U.shape[:-1], 1))
            U = torch.cat([U, U_ones], dim=-1)

            F, U, kl = layer(F, U)
            kl_total += kl

        assert len(kl_total.shape) == 1
        assert kl_total.shape[0] == num_samples
        assert len(F.shape) == 3
        assert F.shape[0] == num_samples
        assert F.shape[2] == self.output_dim

        return F, kl_total

    def ll(self, means, y):
        scales = self.noise * torch.ones_like(means)
        l = torch.distributions.normal.Normal(means, scales)
        return l.log_prob(y).sum()

    def elbo(self, x, y):
        means, kl = self(x)
        ll = self.ll(means, y)
        elbo = ll - kl
        return elbo, ll, kl, self.noise
