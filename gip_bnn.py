import torch
from torch import nn


class GILayer(nn.Module):
    """Represents a single layer of a Bayesian neural network with global inducing points"""

    def __init__(self, input_dim, output_dim, num_induce, nonlinearity=nn.ReLU()):
        super(GILayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_induce = num_induce
        self.nonlinearity = nonlinearity

        # priors
        self.mu_p = torch.zeros(input_dim, output_dim)
        self.logvar_p = torch.zeros(input_dim, output_dim)
        self.prior = torch.distributions.Normal(self.mu_p, (0.5 * self.logvar_p).exp())

        # do we need posterior init here?

        # pseudos
        self.pseud_mu = nn.Parameter(torch.zeros(num_induce, output_dim))
        self.pseud_logprec = nn.Parameter(torch.zeros(num_induce))

    def q_prec(self, U_in):
        psi_U = self.nonlinearity(U_in)
        pseud_prec = torch.exp(self.pseud_logprec)
        
        psi_U_T_lambda_psi_U = torch.einsum("mi, m mj -> ij", psi_U, pseud_prec, psi_U)
        q_prec = torch.diag(torch.exp(-self.logvar_p)[:, 0]) + psi_U_T_lambda_psi_U

    def q_mu(self, U_in, q_prec):
        pass
        
    def forward(self, F_in, U_in):
        pass





class GINetwork(nn.Module):
    """Represents the full network"""