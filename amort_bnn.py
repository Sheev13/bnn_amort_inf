import torch
from torch import nn


class InferenceNetwork(nn.Module):
    """Represents a network that implements amortisation within a layer of the wider network."""

    def __init__(
        self,
        output_dim,
        hidden_dims,
        activation,
    ):
        super(InferenceNetwork, self).__init__()
        self.input_dim = 2
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        self.network = nn.ModuleList()

        self.network.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.network.append(self.activation)

        for i in range(1, len(hidden_dims)):
            self.network.append(
                nn.Linear(
                    self.hidden_dims[i - 1],
                    self.hidden_dims[i],
                )
            )
            self.network.append(self.activation)

        self.network.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        self.network.append(nn.Identity())

    def forward(self, z):
        for layer in self.network:
            z = layer(z)
        return z


class AmortLayer(nn.Module):
    """Represents a single layer of a Bayesian neural network with amortisation."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation,
        prior_var,
        inf_net_dims,
        inf_net_act,
        infer_last_pseudos=True,
    ):
        super(AmortLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.prior_var = prior_var
        self.inf_net_dims = inf_net_dims
        self.inf_net_act = inf_net_act
        self.infer_last_pseudos = infer_last_pseudos

        # priors
        self.mu_p = torch.zeros(output_dim, input_dim)
        self.var_p = self.prior_var * torch.ones(output_dim, input_dim)
        self.full_prior = torch.distributions.MultivariateNormal(
            self.mu_p, self.var_p.diag_embed()
        )

        # amortising/auxiliary inference network
        if self.infer_last_pseudos:
            self.inference_network = InferenceNetwork(
                self.output_dim * 2,
                hidden_dims=self.inf_net_dims,
                activation=self.inf_net_act,
            )

    def infer_pseudos(self, x, y):
        # z is shape (batch_size, 2)
        z = torch.cat((x, y), dim=1)
        # take first output_dim NN outputs as means and last output_dim NN outputs as log stds
        # pseud_mu & pseud_logstd are shape (batch_size, output_dim)
        pseud_mu, pseud_logstd = torch.split(
            self.inference_network(z),
            self.output_dim,
            dim=1,
        )
        pseud_prec = 1 / ((2 * pseud_logstd).exp())
        return pseud_mu.T, pseud_prec.T

    def get_q(self, x, y, F, noise=None) -> torch.distributions.MultivariateNormal:
        # U is shape (num_samples, N, input_dim).
        assert len(F.shape) == 3
        # assert U.shape[1] == self.batch_size
        assert F.shape[2] == self.input_dim

        # U_ is shape (num_samples, 1, batch_size, input_dim).
        F_ = F.unsqueeze(1)

        # amortisation
        if self.infer_last_pseudos:
            pseud_mu, pseud_prec = self.infer_pseudos(x, y)
        else:
            pseud_mu = y.T
            pseud_prec = (1 / (noise**2)) * torch.ones_like(pseud_mu)

        # pseud_prec_ is shape (1, output_dim, 1, batch_size).
        pseud_prec_ = pseud_prec.unsqueeze(0).unsqueeze(-2)

        # pseud_mu_ is shape (1, output_dim, batch_size, 1).
        pseud_mu_ = pseud_mu.unsqueeze(0).unsqueeze(-1)

        # FTL is shape (num_samples, output_dim, input_dim, batch_size)
        FTL = F_.transpose(-1, -2) * pseud_prec_

        # FTLF is shape (num_samples, output_dim, input_dim, input_dim)
        FTLF = FTL @ F_

        # FTLv is shape (num_samples, output_dim, input_dim, 1)
        FTLv = FTL @ pseud_mu_

        # prior_prec_ is shape (1, output_dim, input_dim, input_dim)
        prior_prec_ = (self.var_p ** (-1)).diag_embed().unsqueeze(0)

        q_prec = prior_prec_ + FTLF
        q_prec_chol = torch.linalg.cholesky(q_prec)
        q_cov = torch.cholesky_inverse(q_prec_chol)
        q_mu = (q_cov @ FTLv).squeeze(-1)
        return torch.distributions.MultivariateNormal(q_mu, q_cov)

    def forward(self, x, y, F, F_test=None, noise=None):
        assert len(F.shape) == 3
        assert F.shape[2] == self.input_dim

        if F_test is not None:
            assert len(F_test.shape) == 3
            assert F_test.shape[2] == self.input_dim

        if self.infer_last_pseudos:
            assert noise is None

        q = self.get_q(x, y, F, noise)

        # w should be shape (num_samples, output_dim, input_dim).
        w = q.rsample()

        # kl_contribution is shape (num_samples).
        kl_contribution = torch.distributions.kl.kl_divergence(q, self.full_prior).sum(
            -1
        )

        # F is shape (num_samples, batch_size, output_dim).
        F = self.activation((F @ w.transpose(-1, -2)))

        if F_test is not None:
            F_test = self.activation((F_test @ w.transpose(-1, -2)))

        return F, F_test, kl_contribution


class AmortNetwork(nn.Module):
    """Represents the full Global Inducing Point BNN"""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        nonlinearity=nn.ReLU(),
        prior_var=1.0,
        init_noise=1e-1,
        trainable_noise=True,
        inf_net_dims=[20, 20],
        inf_net_act=nn.ReLU(),
        infer_last_pseudos=False,
    ):
        super(AmortNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity

        self.prior_var = prior_var
        self.log_noise = nn.Parameter(
            torch.tensor(init_noise).log(), requires_grad=trainable_noise
        )

        self.inf_net_dims = inf_net_dims
        self.inf_net_act = inf_net_act
        self.infer_last_pseudos = infer_last_pseudos

        self.network = nn.ModuleList()
        self.network.append(
            AmortLayer(
                self.input_dim + 1,
                self.hidden_dims[0],
                self.nonlinearity,
                self.prior_var,
                self.inf_net_dims,
                self.inf_net_act,
            )
        )
        for i in range(1, len(hidden_dims)):
            self.network.append(
                AmortLayer(
                    self.hidden_dims[i - 1] + 1,
                    self.hidden_dims[i],
                    self.nonlinearity,
                    self.prior_var,
                    self.inf_net_dims,
                    self.inf_net_act,
                )
            )
        self.network.append(
            AmortLayer(
                self.hidden_dims[-1] + 1,
                self.output_dim,
                nn.Identity(),
                self.prior_var,
                self.inf_net_dims,
                self.inf_net_act,
                infer_last_pseudos=self.infer_last_pseudos,
            )
        )

    @property
    def noise(self):
        return torch.exp(self.log_noise)

    def forward(self, x, y, x_test=None, num_samples=1):
        assert len(x.shape) == 2
        assert x.shape[1] == self.input_dim

        # (num_samples, batch_size, input_dim).
        F = x.unsqueeze(0).repeat(num_samples, 1, 1)

        if x_test is not None:
            F_test = x_test.unsqueeze(0).repeat(num_samples, 1, 1)
        else:
            F_test = None

        kl_total = None
        for layer in self.network:
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)

            if F_test is not None:
                F_test_ones = torch.ones(F_test.shape[:-1]).unsqueeze(-1)
                F_test = torch.cat((F_test, F_test_ones), dim=-1)

            if layer.infer_last_pseudos:
                noise = None
            else:
                noise = self.noise

            F, F_test, kl = layer(x, y, F, F_test, noise)

            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        assert len(kl_total.shape) == 1
        assert kl_total.shape[0] == num_samples
        assert len(F.shape) == 3
        assert F.shape[0] == num_samples
        assert F.shape[2] == self.output_dim

        return F, F_test, kl_total

    def ll(self, F, y):
        num_samples = F.shape[0]
        y = y.unsqueeze(0).repeat(num_samples, 1, 1)
        assert y.shape == F.shape

        scales = self.noise * torch.ones_like(F)
        l = torch.distributions.normal.Normal(F, scales)
        log_prob = l.log_prob(y)
        return log_prob.sum(1).sum(1)

    def elbo_loss(self, x, y, num_samples=1):
        F, _, kl = self(x, y, x_test=None, num_samples=num_samples)
        ll = self.ll(F, y)
        assert len(ll.shape) == 1
        assert ll.shape[0] == num_samples
        assert len(kl.shape) == 1
        assert kl.shape[0] == num_samples
        ll = ll.mean()
        kl = kl.mean()
        elbo = ll - kl
        return -elbo, ll, kl, self.noise

    # def get_pseud_outs(self):
    #     if self.infer_last_pseudos:
    #         final_layer = self.network[-1]
    #         outputs = final_layer.infer_pseudos(self.x, self.y)[0].detach().squeeze()
    #     else:
    #         outputs = self.y.squeeze()
    #     return locs, outputs
