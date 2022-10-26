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
