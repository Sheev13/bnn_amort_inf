from typing import Dict, List, Optional

import torch
from torch import nn

from ..utils.activations import NormalActivation
from ..utils.networks import MLP


class AmortisedMFVIBNNLayer(nn.Module):
    """Represents a layer of an amortised factorised-Gaussian BNN"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        input_dim: int,
        output_dim: int,
        in_hidden_dims: List[int],
        in_nonlinearity: nn.Module = nn.ReLU(),
        pw: Optional[torch.distributions.Normal] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pw is None:
            self.pw = torch.distributions.Normal(
                torch.zeros(output_dim, input_dim), torch.ones(output_dim, input_dim)
            )
        else:
            assert isinstance(pw, torch.distributions.Normal)
            assert pw.loc.shape == torch.Size((output_dim, input_dim))
            self.pw = pw

        self._cache: Dict = {}

        self.inference_network = MLP(
            [x_dim + y_dim] + in_hidden_dims + [2 * input_dim * output_dim],
            nonlinearity=in_nonlinearity,
            activation=NormalActivation(),
        )

    @property
    def cache(self):
        return self._cache

    def qw(self, x: torch.Tensor, y: torch.Tensor):
        # w_dist is shape (batch_size, input_dim * output_dim)
        w_dist = self.inference_network(torch.cat((x, y), dim=-1))
        # product over datapoints
        w_dist = w_dist.prod(dim=0)
        # reshape to match prior
        w_dist = w_dist.reshape((self.output_dim, self.input_dim))

        q = self.pw @ w_dist
        assert len(q.shape) == 2
        assert q.shape[0] == self.output_dim
        assert q.shape[1] == self.input_dim

        return q

    def forward(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor):
        pass

    class AmortisedMFVIBNN(nn.Module):
        """Represents the full amortised factorised-Gaussian BNN"""

        def __init__(
            self,
        ):
            super().__init__()
