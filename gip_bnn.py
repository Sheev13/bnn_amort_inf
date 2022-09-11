import torch
from torch import nn


class GILayer(nn.Module):
    """Represents a single layer of a Bayesian neural network with global inducing points"""

    def __init__(self, input_dim, output_dim, num_induce, nonlinearity=nn.ELU()):
        super(GILayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_induce = num_induce
        self.nonlinearity = nonlinearity

        # priors
        self.mu_p = torch.zeros(input_dim, output_dim)
        self.logvar_p = torch.zeros(input_dim, output_dim)
        #TODO: fix this/make it diagonal
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
        
        # what should the shape be?
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
        log_p = self.prior.log_prob(w)
        log_q = q.log_prob(w)
        
        F_out = self.nonlinearity(F_in) @ w  # check that this is the right way around
        U_out = self.nonlinearity(U_in) @ w  # check that this is the right way around
        
        return F_out, U_out, kl_contribution, log_p, log_q
        
        
        
        





class GINetwork(nn.Module):
    """Represents the full network"""