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
        self.mu_p = torch.zeros(input_dim, output_dim)
        self.logvar_p = torch.zeros(input_dim, output_dim)
        #TODO: fix this/make it diagonal
        #TODO: make this a multivariate normal with diagonal covariance so that kl works later on
        self.prior = torch.distributions.Normal(self.mu_p, (0.5 * self.logvar_p).exp())

        # pseudos
        self.pseud_mu = nn.Parameter(torch.zeros(num_induce, output_dim))
        self.pseud_logprec = nn.Parameter(torch.zeros(num_induce))

    def get_q_cov(self, U_in):
        """ returns the covariance matrix of the variational distribution. Uses cholesky decomposition for efficient inversion"""
        psi_U = self.nonlinearity(U_in)
        pseud_prec = torch.exp(self.pseud_logprec)
        
        psi_U_T_lambda_psi_U = torch.einsum("mi, m, mj -> ij", psi_U, pseud_prec, psi_U)  # need to check this
        q_prec = torch.diag(torch.exp(-self.logvar_p)[:, 0]) + psi_U_T_lambda_psi_U  # need to check this
        q_cov = torch.cholesky_inverse(q_prec)
        
        # q_prec_chol = torch.cholesky(q_prec)
        # inv_q_prec_chol = torch.triangular_solve(torch.eye(q_prec_chol.shape[0]), q_prec_chol, upper=False)
        
        # q_cov = inv_q_prec_chol @ inv_q_prec_chol.T
        # q_cov += 1e-5 * torch.eye(q_cov.shape[0])  # for stability?
        # q_cov_chol = torch.cholesky(q_cov)
        
        return q_cov

    def get_q_mu(self, U_in, q_cov):
        psi_U = self.nonlinearity(U_in)
        pseud_prec = torch.exp(self.pseud_logprec)
        
        # what should the shape of this be?
        return q_cov @ psi_U.T @ pseud_prec @ self.pseud_mu  # perhaps use einsum instead?
        
    def forward(self, F_in, U_in):
        # augment inputs with ones to absorb bias
        F_ones = torch.ones(shape=(F_in.shape[0], 1))
        F_in = torch.cat([F_in, F_ones], dim=-1)  # double check dim here
        
        U_ones = torch.ones(shape=(U_in.shape[0], 1))
        U_in = torch.cat([U_in, U_ones], dim=-1) # double check dim here
        
        q_cov = self.get_q_cov(U_in)
        q_mu = self.get_q_mu(U_in, q_cov)
        
        q = torch.distributions.multivariate_normal.MultivariateNormal(q_mu, covariance_matrix=q_cov)
        
        w = q.sample()
        
        kl_contribution = torch.distributions.kl.kl_divergence(q, self.prior).sum()
        
        F_out = self.nonlinearity(F_in) @ w  # check that this is the right way around
        U_out = self.nonlinearity(U_in) @ w  # check that this is the right way around
        
        return F_out, U_out, kl_contribution





class GINetwork(nn.Module):
    """Represents the full Global Inducing Point BNN"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, inducing_points, nonlinearity=nn.ELU()):
        super(GINetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.inducing_points = nn.Parameter(inducing_points)
        self.num_induce = inducing_points.shape[0]
        self.nonlinearity = nonlinearity
        self.log_noise = nn.Parameter(torch.tensor(-1))
        
        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(GILayer(self.input_dim, 
                                            self.hidden_dims[i], 
                                            num_induce=self.num_induce, 
                                            nonlinearity=self.nonlinearity))
            elif i == len(hidden_dims):
                self.network.append(GILayer(self.hidden_dims[i-1],
                                            self.output_dim,
                                            num_induce=self.num_induce,
                                            nonlinearity=self.nonlinearity))
            else:
                self.network.append(GILayer(self.hidden_dims[i-1],
                                            self.hidden_dims[i],
                                            num_induce=self.num_induce,
                                            nonlinearity=self.nonlinearity))
         
    @property            
    def noise(self):
        return torch.exp(self.log_noise)            
                
    def forward(self, F):
        kl_total = torch.tensor(0)
        for i, layer in enumerate(self.network):
            if i == 0:
                F, U, kl = layer(F, self.inducing_points)
            else:
                F, U, kl = layer(F, U)
            kl_total += kl
            
        means = F
        
        return means, self.noise, kl_total
            
    def ll(self, means, y):
        scales = self.noise * torch.ones_like(means)
        l = torch.distributions.normal.Normal(means, scales)
        return l.log_prob(y).sum()
    
    def elbo(self, x, y):
        means, noise, kl = self(x)
        ll = self.ll(means, y)
        elbo = ll - kl
        return elbo, ll, kl, noise
    
    