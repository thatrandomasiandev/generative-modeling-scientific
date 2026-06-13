"""Synthetic 2D Gaussian mixture for diffusion benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gen_sci.data.base import GenerationDataset
from gen_sci.utils.seed import set_seed


@dataclass
class GaussianMixtureDGPConfig:
    """Configuration for a 2D Gaussian mixture DGP."""

    n_samples: int = 5000
    n_components: int = 3
    separation: float = 4.0
    component_std: float = 0.6
    seed: int = 42


def _component_means(n_components: int, separation: float, rng: np.random.Generator) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    radius = separation * (1.0 + 0.15 * rng.standard_normal(n_components))
    means = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return means.astype(np.float64)


def generate_gaussian_mixture_data(config: GaussianMixtureDGPConfig) -> GenerationDataset:
    """Generate samples from a known 2D Gaussian mixture.

    x ~ sum_k pi_k N(mu_k, sigma^2 I), with uniform mixing weights.
    """
    rng = set_seed(config.seed)
    means = _component_means(config.n_components, config.separation, rng)
    weights = np.ones(config.n_components) / config.n_components
    assignments = rng.choice(config.n_components, size=config.n_samples, p=weights)

    samples = np.zeros((config.n_samples, 2), dtype=np.float64)
    for k in range(config.n_components):
        mask = assignments == k
        count = int(mask.sum())
        if count == 0:
            continue
        samples[mask] = rng.normal(
            loc=means[k],
            scale=config.component_std,
            size=(count, 2),
        )

    cov = (config.component_std**2) * np.eye(2)
    return GenerationDataset(
        samples=samples,
        metadata={
            "dgp": "gaussian_mixture",
            "n_components": config.n_components,
            "separation": config.separation,
            "component_std": config.component_std,
            "seed": config.seed,
        },
        ground_truth={
            "means": means,
            "weights": weights,
            "covariance": cov,
            "assignments": assignments,
        },
    )


def sample_gaussian_mixture(config: GaussianMixtureDGPConfig, n_samples: int) -> np.ndarray:
    """Draw fresh samples from the mixture oracle."""
    cfg = GaussianMixtureDGPConfig(
        n_samples=n_samples,
        n_components=config.n_components,
        separation=config.separation,
        component_std=config.component_std,
        seed=config.seed + 999,
    )
    return generate_gaussian_mixture_data(cfg).samples
