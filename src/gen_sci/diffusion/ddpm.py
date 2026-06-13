"""DDPM training and sampling on low-dimensional scientific data.

Provides ``ScoreMLP`` for noise prediction, ``DDPMResult`` as a training
output container, and ``DDPMSampler`` for flexible generation including
DDIM deterministic sampling and classifier-free guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gen_sci.diffusion.schedule import DiffusionSchedule
from gen_sci.utils.device import get_device
from gen_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


class ScoreMLP(nn.Module):
    """Small MLP predicting diffusion noise epsilon.

    Args:
        dim: Dimensionality of the data space.
        hidden: Width of each hidden layer.
        timesteps: Total number of diffusion timesteps (used for
            normalising the time embedding).
    """

    def __init__(self, dim: int, hidden: int = 128, timesteps: int = 100) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy sample ``x`` at timestep ``t``."""
        t_norm = (t.float() / max(self.timesteps - 1, 1)).unsqueeze(-1)
        return self.net(torch.cat([x, t_norm], dim=-1))


class ConditionalScoreMLP(nn.Module):
    """Score network that also accepts a class label ``c``.

    Encodes the label via an embedding table and concatenates it with the
    time-embedded input so that classifier-free guidance can be applied
    at sampling time.

    Args:
        dim: Data dimensionality.
        n_classes: Number of discrete classes (excluding the unconditional
            "null" class, which is mapped to ``n_classes``).
        hidden: Hidden-layer width.
        timesteps: Total diffusion timesteps.
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        hidden: int = 128,
        timesteps: int = 100,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.class_emb = nn.Embedding(n_classes + 1, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise conditioned on class ``c``.

        Args:
            x: Noisy input of shape ``(B, dim)``.
            t: Integer timesteps of shape ``(B,)``.
            c: Class labels of shape ``(B,)``.  When ``None``, uses the
                unconditional embedding (label ``n_classes``).

        Returns:
            Predicted noise of shape ``(B, dim)``.
        """
        t_norm = (t.float() / max(self.timesteps - 1, 1)).unsqueeze(-1)
        if c is None:
            c = torch.full(
                (x.shape[0],), self.n_classes, device=x.device, dtype=torch.long
            )
        c_emb = self.class_emb(c)
        return self.net(torch.cat([x, t_norm, c_emb], dim=-1))


@dataclass
class DDPMResult:
    """Result container for DDPM training.

    Attributes:
        model: Trained noise-prediction network.
        schedule: Diffusion schedule used during training.
        train_loss: Mean MSE loss averaged over the last five epochs.
        device: Device string where the model resides.
    """

    model: ScoreMLP | ConditionalScoreMLP
    schedule: DiffusionSchedule
    train_loss: float
    device: str


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_ddpm(
    samples: np.ndarray,
    timesteps: int = 50,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> DDPMResult:
    """Train a DDPM denoiser on sample data.

    Args:
        samples: Training data of shape ``(N, D)``.
        timesteps: Number of diffusion steps *T*.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden: Hidden-layer width of ``ScoreMLP``.
        seed: Random seed for reproducibility.
        device: Compute device (``"cpu"``, ``"cuda"``, ``"auto"``).

    Returns:
        A ``DDPMResult`` containing the trained model and schedule.
    """
    set_torch_seed(seed)
    dev = get_device(device)
    schedule = DiffusionSchedule.cosine(timesteps)
    model = ScoreMLP(samples.shape[1], hidden=hidden, timesteps=timesteps).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(_to_tensor(samples, dev)),
        batch_size=batch_size,
        shuffle=True,
    )

    alpha_bars = torch.as_tensor(schedule.alpha_bars, dtype=torch.float32, device=dev)
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (x0,) in loader:
            b = x0.shape[0]
            t = torch.randint(0, timesteps, (b,), device=dev)
            eps = torch.randn_like(x0)
            sqrt_ab = torch.sqrt(alpha_bars[t]).unsqueeze(-1)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_bars[t]).unsqueeze(-1)
            xt = sqrt_ab * x0 + sqrt_1m_ab * eps
            pred = model(xt, t)
            loss = nn.functional.mse_loss(pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        logger.debug("epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)

    return DDPMResult(
        model=model,
        schedule=schedule,
        train_loss=float(np.mean(losses[-5:])),
        device=str(dev),
    )


@torch.no_grad()
def sample_ddpm(
    result: DDPMResult,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate samples via ancestral DDPM sampling.

    Args:
        result: Trained ``DDPMResult`` from ``train_ddpm``.
        n_samples: Number of samples to draw.
        seed: Random seed for the sampling noise.

    Returns:
        Array of shape ``(n_samples, D)`` with generated data.
    """
    set_torch_seed(seed)
    dev = torch.device(result.device)
    model = result.model.to(dev)
    model.eval()
    schedule = result.schedule
    dim = model.net[0].in_features - 1
    x = torch.randn(n_samples, dim, device=dev)

    for t in reversed(range(len(schedule.betas))):
        t_batch = torch.full((n_samples,), t, device=dev, dtype=torch.long)
        eps = model(x, t_batch)
        beta = schedule.betas[t]
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        coef1 = 1.0 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1.0 - alpha_bar)
        mean = coef1 * (x - coef2 * eps)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + np.sqrt(beta) * noise
        else:
            x = mean
    return x.cpu().numpy()


class DDPMSampler:
    """Flexible sampler wrapping a trained DDPM for various generation strategies.

    Supports standard ancestral sampling, deterministic DDIM, and
    classifier-free guidance.

    Args:
        result: A ``DDPMResult`` produced by ``train_ddpm``.
    """

    def __init__(self, result: DDPMResult) -> None:
        self.result = result
        self.schedule = result.schedule
        self.device = torch.device(result.device)
        self.model = result.model.to(self.device)

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        alpha_bar_t: float,
    ) -> torch.Tensor:
        """Recover the predicted clean sample x_0 from the noise prediction.

        Math:
            x_0_hat = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)

        Applies dynamic range clipping to prevent explosion when
        ``alpha_bar_t`` is near zero (late diffusion steps).

        Args:
            x_t: Current noisy sample.
            eps_pred: Predicted noise.
            alpha_bar_t: Cumulative alpha at timestep *t*.

        Returns:
            Predicted denoised sample.
        """
        x0 = (x_t - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / max(
            np.sqrt(alpha_bar_t), 1e-6
        )
        return torch.clamp(x0, -10.0, 10.0)

    @torch.no_grad()
    def ddim_sample(
        self,
        n_samples: int,
        seed: int = 0,
    ) -> np.ndarray:
        """Deterministic DDIM sampling (eta=0).

        Uses the reparameterised reverse step:

        .. math::
            x_{t-1} = \\sqrt{\\bar\\alpha_{t-1}}\\, \\hat x_0
                     + \\sqrt{1 - \\bar\\alpha_{t-1}}\\, \\epsilon_\\theta(x_t, t)

        where ``hat x_0`` is the one-step denoised prediction. Because eta=0
        there is no stochastic noise, making the trajectory deterministic given
        the initial latent.

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed (only affects initial noise draw).

        Returns:
            Array of shape ``(n_samples, D)``.
        """
        set_torch_seed(seed)
        self.model.eval()
        dim = self.model.net[0].in_features - 1
        x = torch.randn(n_samples, dim, device=self.device)
        T = len(self.schedule.betas)

        for t in reversed(range(T)):
            t_batch = torch.full(
                (n_samples,), t, device=self.device, dtype=torch.long
            )
            eps_pred = self.model(x, t_batch)
            alpha_bar_t = float(self.schedule.alpha_bars[t])

            x0_pred = self._predict_x0(x, eps_pred, alpha_bar_t)

            if t > 0:
                alpha_bar_prev = float(self.schedule.alpha_bars[t - 1])
                dir_xt = np.sqrt(1.0 - alpha_bar_prev) * eps_pred
                x = np.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            else:
                x = x0_pred

        return x.cpu().numpy()

    @torch.no_grad()
    def guided_sample(
        self,
        n_samples: int,
        labels: torch.Tensor,
        guidance_scale: float = 3.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Classifier-free guided sampling.

        Requires that the underlying model is a ``ConditionalScoreMLP``.
        Combines unconditional and conditional noise predictions:

        .. math::
            \\epsilon_{\\text{guided}} = \\epsilon_{\\text{uncond}}
                + w \\,(\\epsilon_{\\text{cond}} - \\epsilon_{\\text{uncond}})

        Args:
            n_samples: Number of samples to generate.
            labels: Integer class labels of shape ``(n_samples,)``.
            guidance_scale: Weight *w*; values > 1 sharpen conditioning.
            seed: Random seed for initial noise.

        Returns:
            Array of shape ``(n_samples, D)``.

        Raises:
            TypeError: If the model is not a ``ConditionalScoreMLP``.
        """
        if not isinstance(self.model, ConditionalScoreMLP):
            raise TypeError(
                "guided_sample requires a ConditionalScoreMLP model"
            )
        set_torch_seed(seed)
        self.model.eval()

        input_dim = self.model.net[0].in_features
        hidden_dim = self.model.class_emb.embedding_dim
        dim = input_dim - 1 - hidden_dim

        x = torch.randn(n_samples, dim, device=self.device)
        labels = labels.to(self.device)
        T = len(self.schedule.betas)

        for t in reversed(range(T)):
            t_batch = torch.full(
                (n_samples,), t, device=self.device, dtype=torch.long
            )
            eps_cond = self.model(x, t_batch, c=labels)
            eps_uncond = self.model(x, t_batch, c=None)
            eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta = self.schedule.betas[t]
            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alpha_bars[t]
            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x - coef2 * eps_guided)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + np.sqrt(beta) * noise
            else:
                x = mean

        return x.cpu().numpy()
