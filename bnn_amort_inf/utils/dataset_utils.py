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
    noise_std: float = 3.0,
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
    noise_std: float = 0.05,
    min_n: int = 60,
    max_n: int = 120,
    lower: float = -3.0,
    upper: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    noise_std = torch.tensor(noise_std)

    dataset_size = torch.randint(min_n, max_n, (1,))
    period = torch.zeros(1).uniform_(0.8, 1.3)
    gradient = torch.tensor(2.0 / period)

    x = torch.zeros(dataset_size).uniform_(lower, upper)

    sawtooth = lambda x: gradient * (x % period) - 1

    y = sawtooth(x) + noise_std * torch.normal(
        torch.zeros(dataset_size), torch.ones(dataset_size)
    )

    return x.unsqueeze(-1), y.unsqueeze(-1)


def random_mask(ratio: float, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = image.shape[-2:]
    mask = (torch.Tensor(dims).uniform_() < ratio).double()
    return image * mask, mask


def vis_ctxt_img(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    assert len(image.shape) == 3
    colours = image.shape[0]
    assert colours in [1, 3]  # either greyscale or RGB

    mask = mask.bool()
    if colours == 1:
        image = torch.cat((image, image, image), dim=0)
        blue = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(0),
                torch.zeros_like(mask).unsqueeze(0),
                torch.ones_like(mask).unsqueeze(0),
            ),
            dim=0,
        )
        image = torch.where(mask, image, blue)
    elif colours == 3:
        grey = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(0),
                torch.zeros_like(mask).unsqueeze(0),
                torch.ones_like(mask).unsqueeze(0),
            ),
            dim=0,
        )
        image = torch.where(mask, image, grey)

    return image.permute(1, 2, 0)  # permutation needed for matplotlib


def img_for_reg(
    I: torch.Tensor, M_c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_x1, num_x2 = I.shape[-2:]
    num_pix = num_x1 * num_x2
    x1_range = torch.linspace(-1, 1, num_x1)
    x2_range = torch.linspace(-1, 1, num_x2)
    xm1, xm2 = torch.meshgrid(x1_range, x2_range, indexing="xy")
    x1 = xm1.reshape((num_pix))
    x2 = xm2.reshape((num_pix))
    x_t = torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1)
    y_t = I.reshape((I.shape[0], num_pix))
    flat_mc = M_c.reshape((num_pix))
    x_c = x_t[flat_mc.bool()]
    y_c = y_t[:, flat_mc.bool()]
    return x_c, y_c.T, x_t, y_t.T


def samps_to_img_dist(preds: torch.Tensor, img_shape=(28, 28, 1)):
    preds = preds.detach()
    mean, std = preds.mean(0), preds.std(0)
    pred_img = mean.reshape(img_shape)
    pred_std = std.reshape(img_shape)
    return pred_img.numpy(), pred_std.numpy()
