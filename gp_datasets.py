import gpytorch
import torch

class GPDataGenerator():
    """Represents a box of tricks used to generate toy datasets used as meta-learning tasks"""
    
    def __init__(self):
        self.gp = gpytorch.models.ExactGP(
            torch.Tensor(0.0),
            torch.Tensor(0.0),
            gpytorch.likelihoods.GaussianLikelihood(),
        )
        
    def generate_tasks(
        self,
        num_tasks,
    ):
        pass

