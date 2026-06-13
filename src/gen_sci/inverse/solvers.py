"""Inverse problem solvers: classical and diffusion-guided."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso

from gen_sci.diffusion.ddpm import DDPMResult, sample_ddpm, train_ddpm


@dataclass
class InverseResult:
    """Reconstruction result for an inverse problem solver."""

    x_hat: np.ndarray
    method: str
    metadata: dict[str, float | str]


def tikhonov_solve(
    A: np.ndarray,
    y: np.ndarray,
    lam: float = 0.1,
) -> InverseResult:
    """Tikhonov-regularized least squares: min ||Ax - y||^2 + lam ||x||^2."""
    ata = A.T @ A
    rhs = A.T @ y
    x_hat = np.linalg.solve(ata + lam * np.eye(A.shape[1]), rhs)
    return InverseResult(x_hat=x_hat, method="tikhonov", metadata={"lambda": lam})


def lasso_solve(
    A: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
) -> InverseResult:
    """LASSO sparse recovery via coordinate descent."""
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
    model.fit(A, y)
    return InverseResult(x_hat=model.coef_, method="lasso", metadata={"alpha": alpha})


def least_squares_solve(A: np.ndarray, y: np.ndarray) -> InverseResult:
    """Unregularized minimum-norm least squares."""
    x_hat, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return InverseResult(x_hat=x_hat, method="least_squares", metadata={})


def diffusion_posterior_sample(
    A: np.ndarray,
    y: np.ndarray,
    train_signals: np.ndarray,
    n_samples: int = 32,
    guidance_scale: float = 2.0,
    timesteps: int = 30,
    epochs: int = 60,
    seed: int = 42,
    device: str = "cpu",
) -> InverseResult:
    """Diffusion posterior sampling with linear measurement guidance.

    Trains a DDPM prior on signal corpus, then selects the sample with lowest
    measurement residual ||Ax - y|| (approximate posterior mode).
    """
    ddpm = train_ddpm(
        train_signals,
        timesteps=timesteps,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    candidates = sample_ddpm(ddpm, n_samples=n_samples, seed=seed + 1)
    residuals = np.linalg.norm(candidates @ A.T - y, axis=1)
    weights = np.exp(-guidance_scale * residuals)
    weights /= weights.sum() + 1e-12
    x_hat = np.average(candidates, axis=0, weights=weights)
    best_idx = int(np.argmin(residuals))
    return InverseResult(
        x_hat=x_hat,
        method="diffusion_posterior",
        metadata={
            "best_residual": float(residuals[best_idx]),
            "train_loss": ddpm.train_loss,
            "n_candidates": float(n_samples),
        },
    )


SOLVERS = {
    "least_squares": least_squares_solve,
    "tikhonov": tikhonov_solve,
    "lasso": lasso_solve,
}


def solve_inverse(
    A: np.ndarray,
    y: np.ndarray,
    method: str = "tikhonov",
    **kwargs,
) -> InverseResult:
    """Dispatch to a classical inverse problem solver."""
    if method not in SOLVERS:
        raise ValueError(f"Unknown solver: {method}. Choose from {list(SOLVERS)}")
    return SOLVERS[method](A, y, **kwargs)
