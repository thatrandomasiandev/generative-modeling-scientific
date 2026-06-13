"""Molecular design search algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gen_sci.data.molecular_dgp import property_oracle
from gen_sci.diffusion.ddpm import sample_ddpm, train_ddpm


@dataclass
class SearchResult:
    """Result from a molecular design search run."""

    best_latent: np.ndarray
    best_property: float
    all_properties: np.ndarray
    method: str
    n_evaluations: int


def random_search(
    latent_dim: int,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    seed: int = 42,
) -> SearchResult:
    """Uniform random search in latent space."""
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n_evaluations, latent_dim))
    props = property_oracle(latents, optimum, active_dims, interaction_strength)
    best_idx = int(np.argmax(props))
    return SearchResult(
        best_latent=latents[best_idx],
        best_property=float(props[best_idx]),
        all_properties=props,
        method="random",
        n_evaluations=n_evaluations,
    )


def hill_climb_search(
    latent_dim: int,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    step_size: float = 0.3,
    seed: int = 42,
) -> SearchResult:
    """Greedy local search with Gaussian perturbations."""
    rng = np.random.default_rng(seed)
    current = rng.standard_normal(latent_dim)
    current_prop = float(
        property_oracle(current[None], optimum, active_dims, interaction_strength)[0]
    )
    props = [current_prop]
    latents = [current.copy()]

    for _ in range(n_evaluations - 1):
        proposal = current + step_size * rng.standard_normal(latent_dim)
        prop = float(property_oracle(proposal[None], optimum, active_dims, interaction_strength)[0])
        if prop >= current_prop:
            current, current_prop = proposal, prop
        props.append(prop)
        latents.append(current.copy())

    props_arr = np.array(props)
    best_idx = int(np.argmax(props_arr))
    return SearchResult(
        best_latent=latents[best_idx],
        best_property=float(props_arr[best_idx]),
        all_properties=props_arr,
        method="hill_climb",
        n_evaluations=n_evaluations,
    )


def guided_diffusion_search(
    train_latents: np.ndarray,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    timesteps: int = 40,
    epochs: int = 60,
    seed: int = 42,
    device: str = "cpu",
) -> SearchResult:
    """Train latent diffusion prior, sample candidates, rank by oracle property."""
    ddpm = train_ddpm(
        train_latents,
        timesteps=timesteps,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    candidates = sample_ddpm(ddpm, n_samples=n_evaluations, seed=seed + 1)
    props = property_oracle(candidates, optimum, active_dims, interaction_strength)
    best_idx = int(np.argmax(props))
    return SearchResult(
        best_latent=candidates[best_idx],
        best_property=float(props[best_idx]),
        all_properties=props,
        method="guided_diffusion",
        n_evaluations=n_evaluations,
    )


SEARCH_METHODS = {
    "random": random_search,
    "hill_climb": hill_climb_search,
    "guided_diffusion": guided_diffusion_search,
}
