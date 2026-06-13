"""Metrics for molecular design benchmarks."""

from __future__ import annotations

import numpy as np


def top_k_hit_rate(properties: np.ndarray, threshold: float, k: int = 10) -> float:
    """Fraction of top-k discovered molecules above a property threshold."""
    if len(properties) == 0:
        return 0.0
    top_k = np.sort(properties)[-k:]
    return float(np.mean(top_k >= threshold))


def regret(best_found: float, oracle_value: float) -> float:
    """Simple regret: oracle - best found (non-negative)."""
    return float(max(oracle_value - best_found, 0.0))


def normalized_score(best_found: float, random_baseline: float, oracle_value: float) -> float:
    """D4RL-style normalized improvement over random baseline."""
    denom = oracle_value - random_baseline
    if abs(denom) < 1e-12:
        return 0.0
    return float((best_found - random_baseline) / denom)


def latent_diversity(latents: np.ndarray, n_pairs: int = 200, seed: int = 0) -> float:
    """Mean pairwise distance among a random subset of latents."""
    rng = np.random.default_rng(seed)
    if latents.shape[0] < 2:
        return 0.0
    idx = rng.choice(latents.shape[0], size=min(n_pairs, latents.shape[0]), replace=False)
    subset = latents[idx]
    dists = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            dists.append(np.linalg.norm(subset[i] - subset[j]))
    return float(np.mean(dists)) if dists else 0.0
