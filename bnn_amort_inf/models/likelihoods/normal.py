import torch
from torch import nn

from .base import Likelihood


class NormalLikelihood(Likelihood):
    def __init__(self, noise: float, train_noise: bool = True, image: bool = False):
        super().__init__()

        self.log_noise = nn.Parameter(
            torch.as_tensor(noise).log(), requires_grad=train_noise
        )
        self.image = image
        self.soft_clamp = nn.Sigmoid()

    @property
    def noise(self):
        return self.log_noise.exp()

    @noise.setter
    def noise(self, value: float):
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        if self.image:
            x = self.soft_clamp(x)
        return torch.distributions.Normal(x, self.noise)


class HeteroscedasticNormalLikelihood(Likelihood):
    def __init__(self, image: bool = False):
        super().__init__()
        self.image = image
        self.soft_clamp = nn.Sigmoid()
        self.out_dim_multiplier = 2

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.distributions.Normal:
        assert x.shape[-1] % 2 == 0
        loc, log_sigma = torch.chunk(x, chunks=2, dim=dim)
        if self.image:
            loc = self.soft_clamp(loc)
        return torch.distributions.Normal(loc, log_sigma.exp())
