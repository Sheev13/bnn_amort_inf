import numpy as np
import torch
from torch import nn


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.
    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.
    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = nn.functional.pad(t2, (int(padding / 2), int(padding / 2)), "reflect")
        else:  # Odd difference
            t2 = nn.functional.pad(
                t2, (int((padding - 1) / 2), int((padding + 1) / 2)), "reflect"
            )
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = nn.functional.pad(t1, (int(padding / 2), int(padding / 2)), "reflect")
        else:  # Odd difference
            t1 = nn.functional.pad(
                t1, (int((padding - 1) / 2), int((padding + 1) / 2)), "reflect"
            )

    return torch.cat([t1, t2], dim=1)


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.
    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.
    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, "weight"):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, bias)
    return model


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.
    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.
    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.
    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.
    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    return (x - y.permute(0, 2, 1)) ** 2


class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, out_channels, init_length_scale):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 2
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(
            np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True
        )
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales**2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Compute the extra density channel.
        # Shape: (batch, n_in, 1).
        density = torch.ones(batch_size, n_in, 1)

        # Concatenate the channel.
        # Shape: (batch, n_in, in_channels + 1).
        y_out = torch.cat([density, y], dim=2)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels + 1).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels + 1).
        y_out = y_out.sum(1)

        # Use density channel to normalize convolution.
        density, conv = y_out[..., :1], y_out[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out


class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels, init_length_scale):
        super(FinalLayer, self).__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(
            np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True
        )
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales**2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, rho, points_per_unit):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2**self.rho.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        # Instantiate encoder
        self.encoder = ConvDeepSet(
            out_channels=self.rho.in_channels, init_length_scale=init_length_scale
        )

        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(
            in_channels=self.rho.out_channels, init_length_scale=init_length_scale
        )
        self.sigma_layer = FinalLayer(
            in_channels=self.rho.out_channels, init_length_scale=init_length_scale
        )

    def forward(self, x, y, x_out):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).

        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = (
            min(torch.min(x).cpu().numpy(), torch.min(x_out).cpu().numpy(), -2.0) - 0.1
        )
        x_max = (
            max(torch.max(x).cpu().numpy(), torch.max(x_out).cpu().numpy(), 2.0) + 0.1
        )
        num_points = int(
            to_multiple(self.points_per_unit * (x_max - x_min), self.multiplier)
        )
        x_grid = torch.linspace(x_min, x_max, num_points)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.encoder(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError("Shape changed.")

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        return torch.distributions.Normal(mean, sigma)

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod() for param in self.parameters()])

    def npml_loss(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        *args,
        **kwargs,
    ):
        dist = self.forward(x_c.unsqueeze(0), y_c.unsqueeze(0), x_t.unsqueeze(0))
        ll = dist.log_prob(y_t).sum()

        metrics = {"ll": ll.item()}
        return (-ll / x_t.shape[0]), metrics


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.
    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.l2 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=2 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.l3 = nn.Conv1d(
            in_channels=2 * self.in_channels,
            out_channels=2 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.l4 = nn.Conv1d(
            in_channels=2 * self.in_channels,
            out_channels=4 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.l5 = nn.Conv1d(
            in_channels=4 * self.in_channels,
            out_channels=4 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.l6 = nn.Conv1d(
            in_channels=4 * self.in_channels,
            out_channels=8 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(
            in_channels=8 * self.in_channels,
            out_channels=4 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.l8 = nn.ConvTranspose1d(
            in_channels=8 * self.in_channels,
            out_channels=4 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.l9 = nn.ConvTranspose1d(
            in_channels=8 * self.in_channels,
            out_channels=2 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.l10 = nn.ConvTranspose1d(
            in_channels=4 * self.in_channels,
            out_channels=2 * self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.l11 = nn.ConvTranspose1d(
            in_channels=4 * self.in_channels,
            out_channels=self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.l12 = nn.ConvTranspose1d(
            in_channels=2 * self.in_channels,
            out_channels=self.in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.
        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.
        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)
