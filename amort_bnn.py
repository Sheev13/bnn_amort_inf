import torch
from torch import nn

class InferenceNetwork(nn.Module):
    """Represents a network that implements amortisation within a layer of the wider network."""
    
    def __init__(
        self,
        output_dim,
        hidden_dims=[100, 100],
        activation=nn.ReLU(),
    ):
        super(InferenceNetwork, self).__init__()
        self.input_dim = 2
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        
        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(
                    nn.Linear(
                        self.input_dim,
                        self.hidden_dims[i],
                    )
                )
                self.network.append(self.activation)
            elif i == len(hidden_dims):
                self.network.append(
                    nn.Linear(
                        self.hidden_dims[i - 1],
                        self.output_dim,
                    )
                )
                self.network.append(nn.Identity())
            else:
                self.network.append(
                    nn.Linear(
                        self.hidden_dims[i - 1],
                        self.hidden_dims[i],
                    )
                )
                self.network.append(self.activation)
                
    def forward(self, U):
        for layer in self.network:
            U = layer(U)
        return U



class AmortLayer(nn.Module):
    """Represents a single layer of a Bayesian neural network with amortisation."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation,
        prior_var,
    ):
        super(AmortLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.prior_var = prior_var

        # priors
        self.mu_p = torch.zeros(output_dim, input_dim)
        self.var_p = self.prior_var * torch.ones(output_dim, input_dim)
        self.full_prior = torch.distributions.MultivariateNormal(
            self.mu_p, self.var_p.diag_embed()
        )
        
        # auxiliary network
        self.aux_network = InferenceNetwork(self.output_dim * 2)

    def get_q(self, U: torch.Tensor) -> torch.distributions.MultivariateNormal:
        # U is shape (num_samples, num_induces, input_dim).
        assert len(U.shape) == 3
        # assert U.shape[1] == self.num_induce
        assert U.shape[2] == self.input_dim

        # U_ is shape (num_samples, 1, num_induce, input_dim).
        U_ = U.unsqueeze(1)

        # amortisation here: will need to sort this out
        pseud_logprec, pseud_mu = self.aux_network(U)
        
        # what shape are pseud_logprec and pseud_mu?
        
        # pseud_prec_ is shape (1, output_dim, 1, num_induce).
        pseud_prec_ = pseud_logprec.exp().unsqueeze(0).unsqueeze(-2)

        # pseud_mu_ is shape (1, output_dim, num_induce, 1).
        pseud_mu_ = pseud_mu.unsqueeze(0).unsqueeze(-1)

        # UTL is shape (num_samples, output_dim, input_dim, num_induce)
        UTL = U_.transpose(-1, -2) * pseud_prec_

        # UTLU is shape (num_samples, output_dim, input_dim, input_dim)
        UTLU = UTL @ U_

        # UTLv is shape (num_samples, output_dim, input_dim, 1)
        UTLv = UTL @ pseud_mu_

        # prior_prec_ is shape (1, output_dim, input_dim, input_dim)
        prior_prec_ = (self.var_p ** (-1)).diag_embed().unsqueeze(0)

        q_prec = prior_prec_ + UTLU
        q_prec_chol = torch.linalg.cholesky(q_prec)
        q_cov = torch.cholesky_inverse(q_prec_chol)
        q_mu = (q_cov @ UTLv).squeeze(-1)
        return torch.distributions.MultivariateNormal(q_mu, q_cov)

    def forward(self, U):
        assert len(U.shape) == 3
        assert U.shape[2] == self.input_dim

        q = self.get_q(U)

        # w should be shape (num_samples, output_dim, input_dim).
        w = q.rsample()

        # kl_contribution is shape (num_samples).
        kl_contribution = torch.distributions.kl.kl_divergence(q, self.full_prior).sum(
            -1
        )

        # U is shape (num_samples, batch_size, output_dim).
        U = self.activation((U @ w.transpose(-1, -2)))

        return U, kl_contribution


class AmortNetwork(nn.Module):
    """Represents the full Global Inducing Point BNN"""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        inducing_points,
        nonlinearity=nn.ELU(),
        prior_var=1.0,
        init_noise=1e-1,
        trainable_noise=True,
    ):
        super(AmortNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.inducing_points = nn.Parameter(inducing_points)
        self.num_induce = inducing_points.shape[0]
        self.nonlinearity = nonlinearity
        self.prior_var = prior_var
        self.log_noise = nn.Parameter(
            torch.tensor(init_noise).log(), requires_grad=trainable_noise
        )

        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(
                    AmortLayer(
                        self.input_dim + 1,
                        self.hidden_dims[i],
                        num_induce=self.num_induce,
                        activation=self.nonlinearity,
                        prior_var=self.prior_var,
                    )
                )
            elif i == len(hidden_dims):
                self.network.append(
                    AmortLayer(
                        self.hidden_dims[i - 1] + 1,
                        self.output_dim,
                        num_induce=self.num_induce,
                        activation=nn.Identity(),
                        prior_var=self.prior_var,
                    )
                )
            else:
                self.network.append(
                    AmortLayer(
                        self.hidden_dims[i - 1] + 1,
                        self.hidden_dims[i],
                        num_induce=self.num_induce,
                        activation=self.nonlinearity,
                        prior_var=self.prior_var,
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

        kl_total = None
        for layer in self.network:
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)
            U_ones = torch.ones(U.shape[:-1]).unsqueeze(-1)
            U = torch.cat((U, U_ones), dim=-1)

            F, U, kl = layer(F, U)

            if kl_total is None:
                kl_total = kl
            else:
                kl_total += kl

        assert len(kl_total.shape) == 1
        assert kl_total.shape[0] == num_samples
        assert len(F.shape) == 3
        assert F.shape[0] == num_samples
        assert F.shape[2] == self.output_dim

        return F, kl_total

    def ll(self, F, y):
        num_samples = F.shape[0]
        y = y.unsqueeze(0).repeat(num_samples, 1, 1)
        assert y.shape == F.shape

        scales = self.noise * torch.ones_like(F)
        l = torch.distributions.normal.Normal(F, scales)
        log_prob = l.log_prob(y)
        return log_prob.sum(1).sum(1)

    def elbo_loss(self, x, y, num_samples=1):
        F, kl = self(x, num_samples=num_samples)
        ll = self.ll(F, y)
        assert len(ll.shape) == 1
        assert ll.shape[0] == num_samples
        assert len(kl.shape) == 1
        assert kl.shape[0] == num_samples
        ll = ll.mean()
        kl = kl.mean()
        elbo = ll - kl
        return -elbo, ll, kl, self.noise
    
    def get_pseudos(self):
        locs = self.inducing_points.detach().squeeze()
        final_layer = self.network[-1]
        outputs = final_layer.pseud_mu.detach().squeeze()
        return locs, outputs