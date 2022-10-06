import gpytorch
import torch
import numpy as np

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
        noise=0.08,
        min_context=3,
        max_context=50,
        min_target=3,
        max_target=50,
        range=[-2.0, 2.0],
    ):
        task = {'x': torch.linspace(range[0], range[1], 200),
                'y': torch.Tensor(),
                'x_context': torch.Tensor(),
                'y_context': torch.Tensor(),
                'x_target': torch.Tensor(),
                'y_target': torch.Tensor()}
        
        self.gp.eval()
        with gpytorch.settings.prior_mode(True):
            task['y'] = self.gp(task['x']).sample()
        
        num_context = np.random.randint(min_context, max_context+1)
        num_target = np.random.randint(min_target, max_target+1)
        num_points = num_context + num_target
        points_id = torch.randperm(200)[:num_points]
        context_id = points_id[:num_context]
        target_id = points_id[num_context:]
        
        task['x_context'] = task['x'][context_id] + noise*torch.randn(num_context)
        task['y_context'] = task['y'][context_id] + noise*torch.randn(num_context)
        task['x_target'] = task['x'][target_id] + noise*torch.randn(num_target)
        task['y_target'] = task['y'][target_id] + noise*torch.randn(num_target)
        
        return task
