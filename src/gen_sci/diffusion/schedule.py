"""Diffusion noise schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionSchedule:
    """DDPM beta schedule with precomputed alpha products."""

    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray

    @classmethod
    def linear(cls, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> DiffusionSchedule:
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)

    @classmethod
    def cosine(cls, timesteps: int, s: float = 0.008) -> DiffusionSchedule:
        steps = timesteps + 1
        t = np.linspace(0, timesteps, steps, dtype=np.float64)
        f = np.cos(((t / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bars = f / f[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, 1e-5, 0.999)
        alphas = 1.0 - betas
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars[1:])
