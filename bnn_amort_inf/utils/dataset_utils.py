from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset


class MetaDataset(Dataset):
    def __init__(self, datasets: List[Any]):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]


def context_target_split(
    x: torch.Tensor, y: torch.Tensor, min_context: int = 3, max_context: int = 50
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    assert min_context < max_context

    # Randomly sample context points.
    max_context = min(max_context, len(x) - 1)
    n_context = torch.randint(low=min_context, high=max_context, size=(1,))

    rand_idx_perm = torch.randperm(x.shape[0])
    idx_context = rand_idx_perm[:n_context]
    idx_target = rand_idx_perm[n_context:]

    x_c, y_c = x[idx_context], y[idx_context]
    x_t, y_t = x[idx_target], y[idx_target]

    return ((x_c, y_c), (x_t, y_t))


def cubic_dataset(
    noise_std: float = 4.0,
    dataset_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:

    noise_std = torch.tensor(noise_std)

    x_neg, x_pos = torch.zeros(dataset_size // 2), torch.zeros(dataset_size // 2)
    x_neg, x_pos = x_neg.uniform_(-4, -2), x_pos.uniform_(2, 4)
    x = torch.cat((x_neg, x_pos))

    y = x**3 + noise_std * torch.normal(
        torch.zeros(dataset_size), torch.ones(dataset_size)
    )

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    return x.unsqueeze(-1), y.unsqueeze(-1)


def sawtooth_dataset(
    noise_std: float = 0.15,
    min_n: int = 60,
    max_n: int = 120,
    lower: float = -3.0,
    upper: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    noise_std = torch.tensor(noise_std)

    dataset_size = torch.randint(min_n, max_n, (1,))
    period = torch.zeros(1).uniform_(0.3, 1.0)
    gradient = torch.tensor(2.0 / period)

    x = torch.zeros(dataset_size).uniform_(lower, upper)

    sawtooth = lambda x: gradient * (x % period) - 1

    y = sawtooth(x) + noise_std * torch.normal(
        torch.zeros(dataset_size), torch.ones(dataset_size)
    )

    return x.unsqueeze(-1), y.unsqueeze(-1)
