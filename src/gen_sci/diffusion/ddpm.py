"""DDPM training and sampling on low-dimensional scientific data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gen_sci.diffusion.schedule import DiffusionSchedule
from gen_sci.utils.device import get_device
from gen_sci.utils.seed import set_torch_seed


class ScoreMLP(nn.Module):
    """Small MLP predicting diffusion noise epsilon."""

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
        t_norm = (t.float() / max(self.timesteps - 1, 1)).unsqueeze(-1)
        return self.net(torch.cat([x, t_norm], dim=-1))


@dataclass
class DDPMResult:
    """Result container for DDPM training."""

    model: ScoreMLP
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
    """Train a DDPM denoiser on sample data."""
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
    for _ in range(epochs):
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
        losses.append(epoch_loss / max(n_batches, 1))

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
    """Generate samples via ancestral DDPM sampling."""
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
