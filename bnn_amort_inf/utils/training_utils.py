from collections import defaultdict
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm

from .dataset_utils import MetaDataset, context_target_split


def train_metamodel(
    model,
    dataset: MetaDataset,
    neural_process: bool = False,
    min_context: int = 3,
    max_context: int = 50,
    max_iters: int = 10_000,
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
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
    dataset_iterator = iter(dataloader)

    tracker = defaultdict(list)
    iter_tqdm = tqdm(range(max_iters), desc="iters")
    for iter_idx in iter_tqdm:

        try:
            (x, y) = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataloader)
            (x, y) = next(dataset_iterator)

        # TODO: allow for batching of datasets.
        x = x.squeeze(0)
        y = y.squeeze(0)

        opt.zero_grad()

        if neural_process:
            # Randomly sample context and target points.
            (x_c, y_c), (x_t, y_t) = context_target_split(
                x, y, min_context, max_context
            )
            loss, metrics = model.loss(x_c, y_c, x_t, y_t)
        else:
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
