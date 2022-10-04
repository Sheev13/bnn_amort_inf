import gpytorch
import torch

class GPModel(gpytorch.models.ExactGP):
    """Represents a Gaussian process model"""
    
    def __init__(self,
                 train_x=torch.tensor(0.0),
                 train_y=torch.tensor(0.0),
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    

class GPDataGenerator():
    """Represents a box of tricks used to generate toy datasets used as meta-learning tasks"""
    
    def __init__(self):
        self.gp = GPModel()
        
    def generate_task(
        self,
        min_context=5,
        max_context=50,
        min_target=5,
        max_target=50,
        range=[-2.0, 2.0],
    ):
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': []}

