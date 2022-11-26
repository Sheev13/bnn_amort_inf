from typing import Optional

import gpytorch
import torch


class LaplacianKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def forward(self, x1, x2, **params):
        diff = self.covar_dist(x1, x2, **params)
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.exp(-torch.abs(diff))


class GPModel(gpytorch.models.ExactGP):
    """Represents a Gaussian process model"""

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood(),
        kernel: str = "se",
    ):
        super().__init__(x, y, likelihood)

        period_prior = gpytorch.priors.NormalPrior(5.0, 0.2)
        ls_prior = gpytorch.priors.NormalPrior(5.0, 0.4)

        kernels = {
            "se": gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            "per": gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
            "lap": gpytorch.kernels.ScaleKernel(LaplacianKernel()),
            "saw": gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(
                    period_length_prior=period_prior, lengthscale_prior=ls_prior
                )
            ),
        }

        if kernel not in kernels.keys():
            raise ValueError(f"kernel '{kernel}' not recognised")

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernels[kernel]

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
