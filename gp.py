import gpytorch
import torch

class GPModel(gpytorch.models.ExactGP):
    """Represents a Gaussian process model"""
    
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)