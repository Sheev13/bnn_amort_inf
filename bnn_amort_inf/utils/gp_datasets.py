from typing import Optional, Tuple

import gpytorch
import torch


class GPModel(gpytorch.models.ExactGP):
    """Represents a Gaussian process model"""

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood(),
    ):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def gp_dataset_generator(
    x_min: float = -3.0,
    x_max: float = 3.0,
    min_n: int = 51,
    max_n: int = 200,
    noise: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_min < x_max
    assert min_n < max_n

    gp = GPModel()

    # Randomly sample input points from range.
    n = torch.randint(low=min_n, high=max_n, size=(1,))
    x = torch.rand((n, 1)) * (x_min - x_max) + x_max
    gp.eval()
    y = gp(x).sample().unsqueeze(-1)
    y += torch.randn_like(y) * noise

    return x, y
