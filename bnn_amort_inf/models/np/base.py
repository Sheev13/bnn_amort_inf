from typing import Dict, Tuple

import torch
from torch import nn


class BaseNP(nn.Module):
    """Represents a neural process base class"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        embedded_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedded_dim = embedded_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.decoder(self.encoder(x_c, y_c), x_t)

        # elif self.conv:
        #     decode = self.decoder(self.encoder(x_c, y_c, x_t), x_t)
        # else:
        #     decode = self.decoder(self.encoder(x_c, y_c), x_t)

    def npml_loss(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dist = self.forward(x_c, y_c, x_t)
        ll = dist.log_prob(y_t).sum()

        metrics = {"ll": ll.item()}
        return (-ll / x_t.shape[0]), metrics
