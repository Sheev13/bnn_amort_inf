from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import torch
from tqdm import tqdm

from .dataset_utils import MetaDataset, context_target_split, img_for_reg


def train_metamodel(
    model,
    dataset: MetaDataset,
    loss_fn: str = "loss",
    min_context: int = 3,
    max_context: int = 50,
    max_iters: int = 10_000,
    num_samples: int = 1,
    batch_size: int = 1,
    lr: float = 1e-2,
    es: bool = True,
    min_es_iters: int = 1_000,
    ref_es_iters: int = 300,
    smooth_es_iters: int = 50,
    es_thresh: float = 1e-2,
    gridconv: bool = False,
    image: bool = False,
    man_thresh: Optional[Tuple[str, float]] = None,
) -> Dict[str, List[Any]]:
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataloader)

    tracker = defaultdict(list)
    iter_tqdm = tqdm(range(max_iters), desc="iters")
    for iter_idx in iter_tqdm:
        opt.zero_grad()
        batch_loss = torch.tensor(0.0)
        batch_metrics: Dict = defaultdict(float)
        for _ in range(
            batch_size
        ):  # reimplement this to be vectorised (introduce batch dimensionality)?
            if image:
                try:
                    (img, mask) = next(dataset_iterator)
                except StopIteration:
                    dataset_iterator = iter(dataloader)
                    (img, mask) = next(dataset_iterator)

                img, mask = img.squeeze(0), mask.squeeze(0)

                if gridconv:
                    loss, metrics = getattr(model, f"{loss_fn}")(img, mask)
                elif loss_fn in ["npvi_loss", "npml_loss"]:
                    x_c, y_c, x_t, y_t = img_for_reg(img, mask)
                    loss, metrics = getattr(model, f"{loss_fn}")(x_c, y_c, x_t, y_t)
                else:
                    x_c, y_c, x_t, y_t = img_for_reg(img, mask)
                    loss, metrics = getattr(model, f"{loss_fn}")(
                        torch.cat((x_c, x_t)), torch.cat((y_c, y_t))
                    )

            else:
                try:
                    (x, y) = next(dataset_iterator)
                except StopIteration:
                    dataset_iterator = iter(dataloader)
                    (x, y) = next(dataset_iterator)
                x, y = x.squeeze(0), y.squeeze(0)
                if loss_fn in ["npvi_loss", "npml_loss"]:
                    # Randomly sample context and target points.
                    (x_c, y_c), (x_t, y_t) = context_target_split(
                        x, y, min_context, max_context
                    )
                    loss, metrics = getattr(model, f"{loss_fn}")(
                        x_c, y_c, x_t, y_t, num_samples=num_samples
                    )
                else:
                    loss, metrics = getattr(model, f"{loss_fn}")(x, y, num_samples)

            batch_loss += loss / batch_size
            for key, value in metrics.items():
                batch_metrics[key] += value / batch_size

        batch_loss.backward()
        opt.step()

        if "loss" not in batch_metrics:
            tracker["loss"].append(batch_loss.item())

        for key, value in batch_metrics.items():
            tracker[key].append(value)

        iter_tqdm.set_postfix(batch_metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_loss = sum(tracker["loss"][-smooth_es_iters:]) / smooth_es_iters
            ref_loss = (
                sum(tracker["loss"][-ref_es_iters - smooth_es_iters : -ref_es_iters])
                / smooth_es_iters
            )
            if (
                es
                and ref_loss - curr_loss < abs(es_thresh * ref_loss)
                and man_thresh is None
            ):
                break
        if man_thresh is not None and es:
            if batch_metrics[man_thresh[0]] > man_thresh[1]:
                break

    return tracker


def train_model(
    model,
    dataset: torch.utils.data.Dataset,
    max_iters: int = 10_000,
    batch_size: int = 1,
    lr: float = 1e-2,
    es: bool = True,
    min_es_iters: int = 1_000,
    ref_es_iters: int = 300,
    smooth_es_iters: int = 50,
    es_thresh: float = 1e-2,
) -> Dict[str, List[Any]]:
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size
    )
    dataset_iterator = iter(dataloader)

    tracker = defaultdict(list)
    iter_tqdm = tqdm(range(max_iters), desc="iters")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        try:
            (x, y) = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataloader)
            (x, y) = next(dataset_iterator)

        loss, metrics = model.loss(x, y)

        loss.backward()
        opt.step()

        if "loss" not in metrics:
            tracker["loss"].append(loss.item())

        for key, value in metrics.items():
            tracker[key].append(value)

        iter_tqdm.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_loss = sum(tracker["loss"][-smooth_es_iters:]) / smooth_es_iters
            ref_loss = (
                sum(tracker["loss"][-ref_es_iters - smooth_es_iters : -ref_es_iters])
                / smooth_es_iters
            )
            if es and abs(curr_loss - ref_loss) < abs(es_thresh * ref_loss):
                break

    return tracker


def train_gp(
    gp_model,
    likelihood,
    dataset: torch.utils.data.Dataset,
    dataset_size: int,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    es: bool = True,
    min_es_iters: int = 100,
    ref_es_iters: int = 75,
    smooth_es_iters: int = 50,
    es_thresh: float = 1e-2,
) -> Dict[str, List[Any]]:
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters

    gp_model.train()
    likelihood.train()

    opt = torch.optim.Adam(gp_model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size)
    dataset_iterator = iter(dataloader)

    tracker = defaultdict(list)
    iter_tqdm = tqdm(range(max_iters), desc="iters")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        try:
            (x, y) = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataloader)
            (x, y) = next(dataset_iterator)

        output = gp_model(x.squeeze())
        loss = -mll(output, y.squeeze()).sum()

        loss.backward()
        opt.step()

        tracker["loss"].append(-loss.item())

        iter_tqdm.set_postfix(tracker)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_loss = sum(tracker["loss"][-smooth_es_iters:]) / smooth_es_iters
            ref_loss = (
                sum(tracker["loss"][-ref_es_iters - smooth_es_iters : -ref_es_iters])
                / smooth_es_iters
            )
            if es and abs(curr_loss - ref_loss) < abs(es_thresh * ref_loss):
                break

    return tracker
