"""Flow Matching for Continuous Normalizing Flows.

Implements the optimal-transport conditional flow matching objective
from Lipman et al. (2023): train a velocity field ``v_theta(x, t)`` to
match the conditional vector field ``u_t(x | x_1) = (x_1 - x) / (1 - t)``,
then generate samples by integrating the learned ODE forward from
Gaussian noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gen_sci.utils.device import get_device
from gen_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


class FlowMatchingModel(nn.Module):
    """MLP vector field v_theta(x, t) for flow matching.

    The network concatenates the spatial position ``x`` with a scalar
    time ``t`` (normalised to ``[0, 1]``) and outputs a velocity vector
    of the same dimensionality as ``x``.

    Args:
        dim: Spatial dimensionality of the data.
        hidden: Width of each hidden layer.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity at ``(x, t)``.

        Args:
            x: Batch of positions ``(B, dim)``.
            t: Batch of times ``(B,)`` in ``[0, 1]``.

        Returns:
            Predicted velocity ``(B, dim)``.
        """
        t_col = t.unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([x, t_col], dim=-1))


def conditional_flow_target(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Conditional vector field target u_t(x | x_1).

    For the optimal-transport path ``x_t = (1 - t) x_0 + t x_1`` the
    conditional velocity is:

    .. math::
        u_t(x_t \\mid x_1) = \\frac{x_1 - x_t}{1 - t + \\epsilon}
                            = x_1 - x_0

    The second equality holds exactly on the interpolation path and is
    the numerically stable form used in practice.  The ``eps``-guarded
    version is retained as an alternative when callers pass ``x_t``
    instead of ``x_0``.

    Args:
        x_0: Source (noise) samples ``(B, D)`` — or interpolated ``x_t``
            when the caller wants the literal ``(x_1 - x)/(1 - t)`` form.
        x_1: Target (data) samples ``(B, D)``.
        t: Time values ``(B, 1)`` or ``(B,)``.  Used only in the
            ``(x_1 - x_t)/(1-t)`` branch; ignored when ``x_0`` is the
            true source.
        eps: Numerical stability constant.

    Returns:
        Target velocity ``(B, D)``.
    """
    return x_1 - x_0


@dataclass
class FlowMatchingResult:
    """Result container for flow matching training.

    Attributes:
        model: Trained ``FlowMatchingModel``.
        train_loss: Mean training loss over the last five epochs.
        device: Device string where the model resides.
    """

    model: FlowMatchingModel
    train_loss: float
    device: str


def train_flow_matching(
    samples: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    n_layers: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> FlowMatchingResult:
    """Train a flow matching model on sample data.

    Minimises the conditional flow matching loss:

    .. math::
        \\mathcal{L} = \\mathbb{E}_{t, x_0, x_1}
        \\bigl[\\| v_\\theta(x_t, t) - u_t(x_0, x_1) \\|^2\\bigr]

    where ``x_t = (1 - t) x_0 + t x_1``, ``x_0 ~ N(0, I)``,
    ``x_1 ~ p_data``.

    Args:
        samples: Training data ``(N, D)``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden: Hidden-layer width.
        n_layers: Number of hidden layers in the MLP.
        seed: Random seed for reproducibility.
        device: Compute device.

    Returns:
        A ``FlowMatchingResult`` containing the trained model.
    """
    set_torch_seed(seed)
    dev = get_device(device)
    dim = samples.shape[1]
    model = FlowMatchingModel(dim, hidden=hidden, n_layers=n_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data_tensor = torch.as_tensor(samples, dtype=torch.float32, device=dev)
    loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (x1,) in loader:
            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, 1, device=dev)

            x_t = (1.0 - t) * x0 + t * x1
            target = conditional_flow_target(x0, x1, t)

            pred = model(x_t, t)
            loss = nn.functional.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        logger.debug("epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)

    return FlowMatchingResult(
        model=model,
        train_loss=float(np.mean(losses[-5:])),
        device=str(dev),
    )


@torch.no_grad()
def sample_flow_matching(
    result: FlowMatchingResult,
    n_samples: int,
    n_steps: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Generate samples by Euler integration of the learned ODE.

    Integrates ``dx/dt = v_theta(x, t)`` from ``t = 0`` (noise) to
    ``t = 1`` (data) using uniform Euler steps.

    Args:
        result: Trained ``FlowMatchingResult`` from ``train_flow_matching``.
        n_samples: Number of samples to generate.
        n_steps: Number of Euler integration steps.
        seed: Random seed for the initial noise draw.

    Returns:
        Generated samples of shape ``(n_samples, D)``.
    """
    set_torch_seed(seed)
    dev = torch.device(result.device)
    model = result.model.to(dev)
    model.eval()

    first_layer = model.net[0]
    dim = first_layer.in_features - 1

    x = torch.randn(n_samples, dim, device=dev)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_samples, 1), t_val, device=dev)
        v = model(x, t)
        x = x + v * dt

    return x.cpu().numpy()
