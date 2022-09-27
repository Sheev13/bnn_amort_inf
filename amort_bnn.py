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
        
        # amortising/auxiliary inference network
        self.inference_network = InferenceNetwork(self.output_dim * 2)
        
    def infer_pseudos(self, x, y):
        # z is shape (batch_size, 2)
        z = torch.cat((x, y), dim=1)
        # take first output_dim NN outputs as means and last output_dim NN outputs as log stds
        # pseud_mu & pseud_logstd are shape (batch_size, output_dim)
        pseud_mu, pseud_logstd = torch.split(
            self.inference_network(z),
            self.output_dim,
            dim=2,
        )
        pseud_prec = 1 / ((2 * pseud_logstd).exp())
        return pseud_mu.T, pseud_prec.T
    
    def get_q(self, U: torch.Tensor, x, y) -> torch.distributions.MultivariateNormal:
        # U is shape (num_samples, N, input_dim).
        assert len(U.shape) == 3
        # assert U.shape[1] == self.batch_size
        assert U.shape[2] == self.input_dim

        # U_ is shape (num_samples, 1, batch_size, input_dim).
        U_ = U.unsqueeze(1)

        # amortisation
        pseud_mu, pseud_prec = self.infer_pseudos(x, y)
        
        # pseud_prec_ is shape (1, output_dim, 1, batch_size).
        pseud_prec_ = pseud_prec.unsqueeze(0).unsqueeze(-2)

        # pseud_mu_ is shape (1, output_dim, batch_size, 1).
        pseud_mu_ = pseud_mu.unsqueeze(0).unsqueeze(-1)

        # UTL is shape (num_samples, output_dim, input_dim, batch_size)
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

    def forward(self, F, U, x, y):
        assert len(U.shape) == 3
        assert len(F.shape) == 3
        assert U.shape[2] == self.input_dim
        assert F.shape[2] == self.input_dim

        q = self.get_q(U, x, y)

        # w should be shape (num_samples, output_dim, input_dim).
        w = q.rsample()

        # kl_contribution is shape (num_samples).
        kl_contribution = torch.distributions.kl.kl_divergence(q, self.full_prior).sum(
            -1
        )

        # F is shape (num_samples, batch_size, output_dim).
        F = self.activation((F @ w.transpose(-1, -2)))
        U = self.activation((U @ w.transpose(-1, -2)))

        return F, U, kl_contribution


class AmortNetwork(nn.Module):
    """Represents the full Global Inducing Point BNN"""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        x,
        y,
        nonlinearity=nn.ReLU(),
        prior_var=1.0,
        init_noise=1e-1,
        trainable_noise=True,
    ):
        super(AmortNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.x = x
        self.y = y
        self.nonlinearity = nonlinearity
        self.prior_var = prior_var
        self.log_noise = nn.Parameter(
            torch.tensor(init_noise).log(), requires_grad=trainable_noise
        )

        self.network = nn.ModuleList()
        
        self.network.append(
            AmortLayer(
                self.input_dim + 1,
                self.hidden_dims[0],
                activation=self.nonlinearity,
                prior_var=self.prior_var,
                )
            )
        for i in range(1, len(hidden_dims)):
            self.network.append(
                AmortLayer(
                    self.hidden_dims[i - 1] + 1,
                    self.hidden_dims[i],
                    activation=self.nonlinearity,
                    prior_var=self.prior_var,
                )
                )
        self.network.append(
            AmortLayer(
                self.hidden_dims[-1] + 1,
                self.output_dim,
                activation=nn.Identity(),
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
        U = self.x.unsqueeze(0).repeat(num_samples, 1, 1)

        kl_total = None
        for layer in self.network:
            F_ones = torch.ones(F.shape[:-1]).unsqueeze(-1)
            F = torch.cat((F, F_ones), dim=-1)
            U_ones = torch.ones(U.shape[:-1]).unsqueeze(-1)
            U = torch.cat((U, U_ones), dim=-1)

            F, U, kl = layer(F, U, self.x, self.y)

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
    
    
    # will need to fix this up later
    
    # def get_pseud_outs(self):
    #     locs = self.x.squeeze()
    #     final_layer = self.network[-1]
    #     outputs = final_layer.pseud_mu.detach().squeeze()
    #     return locs, outputs
    
    
class AmortNetworkWrapper(nn.Module):
    """Wrapper for the Amortised Inference Network"""
    
    def __init__(self, amortnetwork, amortlayers, inferencelayers):
        super(AmortNetworkWrapper, self).__init__()
        self.amortnetwork = amortnetwork
        self.amortlayers = amortlayers
        self.inferencelayers = inferencelayers