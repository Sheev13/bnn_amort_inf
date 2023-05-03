from abc import ABC
from typing import Dict, Optional

import torch
from torch import nn


class BaseBNNLayer(nn.Module, ABC):
    """Represents a single layer of a BNN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pw: Optional[torch.distributions.Normal] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pw is None:
            self.pw = torch.distributions.Normal(
                torch.zeros(output_dim, input_dim),
                (torch.ones(output_dim, input_dim)).sqrt(),
            )
        else:
            assert isinstance(pw, torch.distributions.Normal)
            assert pw.loc.shape == torch.Size((output_dim, input_dim))
            self.pw = pw

        self._cache: Dict = {}

    @property
    def cache(self):
        return self._cache

    def forward(self, *args, **kwargs):
        """Does things and stores in self._cache."""
        raise NotImplementedError
