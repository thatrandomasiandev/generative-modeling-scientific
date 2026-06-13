"""Metrics for generative model evaluation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def maximum_mean_discrepancy(
    generated: np.ndarray,
    reference: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Unbiased RBF-kernel MMD between generated and reference samples."""
    if gamma is None:
        combined = np.vstack([generated, reference])
        dists = cdist(combined, combined)
        gamma = 1.0 / max(float(np.median(dists[dists > 0])), 1e-6)

    def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d2 = cdist(x, y, metric="sqeuclidean")
        return np.exp(-gamma * d2)

    n = generated.shape[0]
    m = reference.shape[0]
    k_xx = _kernel(generated, generated)
    k_yy = _kernel(reference, reference)
    k_xy = _kernel(generated, reference)
    mmd = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return float(max(mmd, 0.0))


def mode_coverage(
    generated: np.ndarray,
    means: np.ndarray,
    threshold: float = 1.5,
) -> float:
    """Fraction of mixture modes with at least one nearby generated sample."""
    if means.ndim == 1:
        means = means.reshape(1, -1)
    covered = 0
    for mean in means:
        dists = np.linalg.norm(generated - mean, axis=1)
        if np.any(dists <= threshold):
            covered += 1
    return covered / means.shape[0]


def mean_pairwise_distance(samples: np.ndarray) -> float:
    """Average pairwise L2 distance — diversity proxy."""
    if samples.shape[0] < 2:
        return 0.0
    dists = cdist(samples, samples)
    idx = np.triu_indices(samples.shape[0], k=1)
    return float(dists[idx].mean())
