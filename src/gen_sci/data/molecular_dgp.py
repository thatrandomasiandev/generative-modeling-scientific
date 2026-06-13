"""Synthetic molecular property landscape with known optimum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gen_sci.data.base import MolecularDataset
from gen_sci.utils.seed import set_seed


@dataclass
class MolecularDGPConfig:
    """Configuration for latent molecular design benchmark."""

    n_molecules: int = 5000
    latent_dim: int = 8
    n_active_dims: int = 3
    interaction_strength: float = 1.5
    seed: int = 42


def property_oracle(
    latents: np.ndarray,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
) -> np.ndarray:
    """Compute binding affinity proxy with known global optimum."""
    z = latents[:, active_dims]
    z_star = optimum[active_dims]
    base = -np.sum((z - z_star) ** 2, axis=1)
    if len(active_dims) >= 2:
        cross = z[:, 0] * z[:, 1]
        target = z_star[0] * z_star[1]
        base += interaction_strength * (1.0 - (cross - target) ** 2)
    return base.astype(np.float64)


def generate_molecular_data(config: MolecularDGPConfig) -> MolecularDataset:
    """Generate latent molecules with oracle property scores.

    z ~ N(0, I), property f(z) peaks at a known z* with sparse active subspace.
    """
    rng = set_seed(config.seed)
    latents = rng.standard_normal((config.n_molecules, config.latent_dim))
    active_dims = np.sort(rng.choice(config.latent_dim, size=config.n_active_dims, replace=False))
    optimum = np.zeros(config.latent_dim, dtype=np.float64)
    optimum[active_dims] = rng.uniform(-1.5, 1.5, size=config.n_active_dims)
    properties = property_oracle(
        latents, optimum, active_dims, config.interaction_strength
    )

    return MolecularDataset(
        latents=latents,
        properties=properties,
        metadata={
            "dgp": "latent_molecular_landscape",
            "latent_dim": config.latent_dim,
            "n_active_dims": config.n_active_dims,
            "interaction_strength": config.interaction_strength,
            "seed": config.seed,
        },
        ground_truth={
            "optimum": optimum,
            "active_dims": active_dims,
            "oracle_value": float(property_oracle(optimum[None], optimum, active_dims, config.interaction_strength)[0]),
            "top_1_percent_threshold": float(np.quantile(properties, 0.99)),
        },
    )
