"""Metrics for inverse problem reconstruction."""

from __future__ import annotations

import numpy as np


def relative_l2_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """||x_hat - x|| / ||x||."""
    denom = np.linalg.norm(x_true) + 1e-12
    return float(np.linalg.norm(x_hat - x_true) / denom)


def support_recovery(x_hat: np.ndarray, x_true: np.ndarray, threshold: float = 1e-2) -> float:
    """Jaccard index between supports of x_hat and x_true."""
    true_support = set(np.flatnonzero(np.abs(x_true) > threshold))
    est_support = set(np.flatnonzero(np.abs(x_hat) > threshold))
    if not true_support and not est_support:
        return 1.0
    if not true_support or not est_support:
        return 0.0
    return len(true_support & est_support) / len(true_support | est_support)


def measurement_consistency(A: np.ndarray, x_hat: np.ndarray, y: np.ndarray) -> float:
    """Relative measurement residual ||Ax_hat - y|| / ||y||."""
    denom = np.linalg.norm(y) + 1e-12
    return float(np.linalg.norm(A @ x_hat - y) / denom)


def psnr(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Peak signal-to-noise ratio treating signals as 1D images."""
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse < 1e-12:
        return 100.0
    peak = float(np.max(np.abs(x_true)) + 1e-12)
    return float(20 * np.log10(peak) - 10 * np.log10(mse))
