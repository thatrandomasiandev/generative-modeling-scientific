"""Diffusion noise schedules.

Provides ``DiffusionSchedule`` with linear, cosine, and quadratic beta
schedules plus derived quantities (SNR, importance-weighted timestep
sampling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionSchedule:
    """DDPM beta schedule with precomputed alpha products.

    Attributes:
        betas: Noise schedule beta_t of shape ``(T,)``.
        alphas: ``1 - betas``.
        alpha_bars: Cumulative product of alphas, ``cumprod(alphas)``.
    """

    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray

    @classmethod
    def linear(
        cls,
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> DiffusionSchedule:
        """Linearly-spaced beta schedule.

        Args:
            timesteps: Number of diffusion steps *T*.
            beta_start: Initial (smallest) beta value.
            beta_end: Final (largest) beta value.

        Returns:
            A ``DiffusionSchedule`` with linearly increasing betas.
        """
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)

    @classmethod
    def cosine(cls, timesteps: int, s: float = 0.008) -> DiffusionSchedule:
        """Cosine schedule from Nichol & Dhariwal (2021).

        Args:
            timesteps: Number of diffusion steps *T*.
            s: Small offset preventing singularity near ``t = 0``.

        Returns:
            A ``DiffusionSchedule`` with cosine-derived betas.
        """
        steps = timesteps + 1
        t = np.linspace(0, timesteps, steps, dtype=np.float64)
        f = np.cos(((t / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bars = f / f[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, 1e-5, 0.999)
        alphas = 1.0 - betas
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars[1:])

    @classmethod
    def quadratic(
        cls,
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> DiffusionSchedule:
        """Quadratic beta schedule: beta_t = beta_start + (beta_end - beta_start) * (t/T)^2.

        Grows noise more slowly at early timesteps (where fine detail is
        destroyed) and accelerates toward the end.

        Args:
            timesteps: Number of diffusion steps *T*.
            beta_start: Initial beta value.
            beta_end: Final beta value.

        Returns:
            A ``DiffusionSchedule`` with quadratically increasing betas.
        """
        t = np.arange(timesteps, dtype=np.float64)
        betas = beta_start + (beta_end - beta_start) * (t / max(timesteps - 1, 1)) ** 2
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)

    def snr(self) -> np.ndarray:
        """Signal-to-noise ratio at each timestep.

        Math:
            SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

        Returns:
            Array of shape ``(T,)`` with positive SNR values.
        """
        return self.alpha_bars / (1.0 - self.alpha_bars + 1e-12)

    def optimal_t_sampler(self, seed: int = 0) -> np.ndarray:
        """Importance-sample timesteps proportional to SNR loss weights.

        The loss weight for each timestep in the variational bound is
        proportional to ``SNR(t-1) - SNR(t)`` (the "discrete SNR change").
        This method returns a precomputed probability vector over ``[0, T)``
        so that ``np.random.choice(T, p=probs)`` yields importance-weighted
        timesteps.

        Args:
            seed: Unused (provided for API consistency); the method
                returns deterministic probabilities.

        Returns:
            Probability array of shape ``(T,)`` summing to 1.
        """
        snr_vals = self.snr()
        weights = np.zeros_like(snr_vals)
        weights[0] = snr_vals[0]
        weights[1:] = np.maximum(snr_vals[:-1] - snr_vals[1:], 1e-12)
        weights = np.maximum(weights, 1e-12)
        return weights / weights.sum()
