import sys
from typing import Optional, Tuple

import torch

sys.path.append("../")
from bnn_amort_inf.models.gp import GPModel


def gp_dataset_generator(
    x_min: float = -3.0,
    x_max: float = 3.0,
    min_n: int = 60,
    max_n: int = 120,
    noise: float = 0.01,
    kernel: str = "se",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_min < x_max
    assert min_n < max_n

    gp = GPModel(
        kernel=kernel,
    )

    # Randomly sample input points from range.
    n = torch.randint(low=min_n, high=max_n, size=(1,))
    x = torch.rand((n, 1)) * (x_min - x_max) + x_max
    gp.eval()
    y = gp(x).sample().unsqueeze(-1)
    y += torch.randn_like(y) * noise

    return x, y
