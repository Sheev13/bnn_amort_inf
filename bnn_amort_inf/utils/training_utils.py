import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import torch
from tqdm import tqdm

from npf import CNPFLoss

from .dataset_utils import MetaDataset, context_target_split


def train_metamodel(
    model,
    dataset: MetaDataset,
    neural_process: bool = False,
    np_loss: bool = False,
    np_kl: bool = True,
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
    npf_model: bool = False,
) -> Dict[str, List[Any]]:
    assert ref_es_iters < min_es_iters
    assert smooth_es_iters < min_es_iters

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataloader)

    if npf_model:
        loss_object = CNPFLoss()

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
                    (I, M_c, x_c, y_c, x_t, y_t) = next(dataset_iterator)
                except StopIteration:
                    dataset_iterator = iter(dataloader)
                    (I, M_c, x_c, y_c, x_t, y_t) = next(dataset_iterator)
                I, M_c = I.squeeze(0), M_c.squeeze(0)
                x_c, y_c, x_t, y_t = (
                    x_c.squeeze(0),
                    y_c.squeeze(0),
                    x_t.squeeze(0),
                    y_t.squeeze(0),
                )
                assert len(I.shape) == 3
                if gridconv:
                    loss, metrics = model.loss(I, M_c)
                elif np_loss:
                    loss, metrics = model.np_loss(x_c, y_c, x_t, y_t, kl=np_kl)
                else:
                    loss, metrics = model.loss(x_c, y_c)

            else:
                try:
                    (x, y) = next(dataset_iterator)
                except StopIteration:
                    dataset_iterator = iter(dataloader)
                    (x, y) = next(dataset_iterator)
                x, y = x.squeeze(0), y.squeeze(0)
                if neural_process or np_loss or npf_model:
                    # Randomly sample context and target points.
                    (x_c, y_c), (x_t, y_t) = context_target_split(
                        x, y, min_context, max_context
                    )
                    if npf_model:
                        preds = model(
                            x_c.unsqueeze(0), y_c.unsqueeze(0), x_t.unsqueeze(0)
                        )
                        loss = loss_object.get_loss(
                            preds[0], preds[1], preds[2], preds[3], y_t.unsqueeze(0)
                        )[0]
                        metrics = {"ll": -loss.item()}
                    elif np_loss:
                        loss, metrics = model.np_loss(
                            x_c, y_c, x_t, y_t, num_samples, kl=np_kl
                        )
                    else:
                        loss, metrics = model.loss(x_c, y_c, x_t, y_t)
                else:
                    loss, metrics = model.loss(x, y, num_samples)

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
