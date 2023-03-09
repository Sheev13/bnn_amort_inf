import torch
from torch import nn


class NormalLikelihood(nn.Module):
    def __init__(self, noise: float, train_noise: bool = True):
        super().__init__()

        self.log_noise = nn.Parameter(
            torch.as_tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    @noise.setter
    def noise(self, value: float):
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())

    def forward(self, out: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(out, self.noise)


class BernoulliLikelihood(nn.Module):
    def forward(self, out: torch.Tensor) -> torch.distributions.Bernoulli:
        return torch.distributions.Bernoulli(logits=out)
