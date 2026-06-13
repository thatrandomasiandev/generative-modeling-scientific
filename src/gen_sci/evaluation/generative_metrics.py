"""Standalone generative-model evaluation metrics.

Provides distribution-comparison measures independent of diffusion or
flow-matching internals: Maximum Mean Discrepancy (MMD), the Coverage
& Density pair from Naeem et al. (2020), and exact 1-D Wasserstein
distance on random projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


def mmd(
    generated: np.ndarray,
    reference: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Maximum Mean Discrepancy with an RBF kernel.

    Computes the *unbiased* estimator:

    .. math::
        \\widehat{\\text{MMD}}^2 = \\frac{1}{n(n-1)} \\sum_{i \\neq j} k(x_i, x_j)
            + \\frac{1}{m(m-1)} \\sum_{i \\neq j} k(y_i, y_j)
            - \\frac{2}{nm} \\sum_{i,j} k(x_i, y_j)

    where ``k`` is the Gaussian RBF kernel with bandwidth ``gamma``.

    Args:
        generated: Generated samples ``(N, D)``.
        reference: Reference (ground-truth) samples ``(M, D)``.
        gamma: RBF bandwidth. If ``None``, uses the median heuristic.

    Returns:
        Non-negative MMD^2 estimate (clamped to zero).
    """
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
    val = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return float(max(val, 0.0))


@dataclass
class CoverageDensityResult:
    """Container for coverage and density metrics.

    Attributes:
        coverage: Fraction of reference samples whose nearest generated
            neighbour is within a distance threshold (recall-like).
        density: Fraction of generated samples whose nearest reference
            neighbour is within a distance threshold (precision-like).
    """

    coverage: float
    density: float


def coverage_and_density(
    generated: np.ndarray,
    reference: np.ndarray,
    k: int = 5,
) -> CoverageDensityResult:
    """Coverage and Density metrics (Naeem et al. 2020).

    For each reference point, *coverage* checks whether at least one
    generated sample falls within its ``k``-th nearest-neighbour radius.
    For each generated point, *density* counts how many reference samples
    it covers, normalised by ``k``.

    Args:
        generated: Generated samples ``(N, D)``.
        reference: Reference samples ``(M, D)``.
        k: Number of nearest neighbours used to define the manifold
            radius.  Clamped to ``min(k, M-1, N-1)`` to avoid
            out-of-bounds indexing.

    Returns:
        A ``CoverageDensityResult`` with coverage in ``[0, 1]`` and
        density >= 0.
    """
    n = generated.shape[0]
    m = reference.shape[0]
    k = min(k, m - 1, n - 1)
    if k < 1:
        return CoverageDensityResult(coverage=0.0, density=0.0)

    d_ref_ref = cdist(reference, reference)
    np.fill_diagonal(d_ref_ref, np.inf)
    radii_ref = np.sort(d_ref_ref, axis=1)[:, k - 1]

    d_gen_ref = cdist(generated, reference)

    covered = 0
    for j in range(m):
        if np.any(d_gen_ref[:, j] <= radii_ref[j]):
            covered += 1
    coverage = covered / m

    d_gen_gen = cdist(generated, generated)
    np.fill_diagonal(d_gen_gen, np.inf)
    radii_gen = np.sort(d_gen_gen, axis=1)[:, k - 1]

    density_sum = 0.0
    for i in range(n):
        count = np.sum(d_gen_ref[i, :] <= radii_gen[i])
        density_sum += count / k
    density = density_sum / n

    return CoverageDensityResult(coverage=coverage, density=density)


def wasserstein_1d(
    generated: np.ndarray,
    reference: np.ndarray,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    """Sliced 1-D Wasserstein distance.

    Projects both distributions onto ``n_projections`` random unit
    directions and computes the exact W1 on each 1-D projection
    (which is simply the L1 distance between sorted values).  The
    final result is the mean over all projections.

    Args:
        generated: Generated samples ``(N, D)``.
        reference: Reference samples ``(M, D)``.
        n_projections: Number of random projections.
        seed: Random seed for the projection directions.

    Returns:
        Estimated sliced Wasserstein-1 distance (non-negative).
    """
    rng = np.random.default_rng(seed)
    dim = generated.shape[1]
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    total = 0.0
    for d in directions:
        proj_gen = np.sort(generated @ d)
        proj_ref = np.sort(reference @ d)

        n = len(proj_gen)
        m = len(proj_ref)
        if n != m:
            interp_ref = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, m),
                proj_ref,
            )
            total += float(np.mean(np.abs(proj_gen - interp_ref)))
        else:
            total += float(np.mean(np.abs(proj_gen - proj_ref)))

    return total / n_projections
