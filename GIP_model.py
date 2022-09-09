import torch
from torch import nn

class GILayer(nn.Module):
    """Represents a single layer of the network"""

    def __init__(self, input_dim, output_dim, num_induce):
        super(GILayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_induce = num_induce

        # priors
        self.w_mu_p = torch.zeros(input_dim, output_dim)
        self.w_logvar_p = torch.zeros(input_dim, output_dim)
        self.b_mu_p = torch.zeros(input_dim, output_dim)
        self.b_logvar_p = torch.zeros(input_dim, output_dim)

        # posteriors

        # pseudos
        self.pseud_mu = nn.Parameter(torch.zeros(num_induce, output_dim))
        self.pseud_logprec = nn.Parameter(torch.zeros(num_induce))





class GINetwork(nn.Module):
    """Represents the full network"""