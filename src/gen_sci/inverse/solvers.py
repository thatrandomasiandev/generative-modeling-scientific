"""Inverse problem solvers: classical and diffusion-guided.

Includes Tikhonov, LASSO, vanilla diffusion posterior sampling, Diffusion
Posterior Sampling (DPS; Chung et al. 2023), and Pseudoinverse-Guided
Diffusion Models (PGDM; Song et al. 2023).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import Lasso

from gen_sci.diffusion.ddpm import DDPMResult, sample_ddpm, train_ddpm
from gen_sci.diffusion.schedule import DiffusionSchedule
from gen_sci.utils.device import get_device
from gen_sci.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


@dataclass
class InverseResult:
    """Reconstruction result for an inverse problem solver.

    Attributes:
        x_hat: Recovered signal of shape ``(D,)``.
        method: Name of the solver that produced this result.
        metadata: Solver-specific diagnostic values.
    """

    x_hat: np.ndarray
    method: str
    metadata: dict[str, float | str]


# ---------------------------------------------------------------------------
# Configurable forward operators
# ---------------------------------------------------------------------------

def _measurement_fn(
    A: np.ndarray | torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a differentiable forward-measurement operator y = A x.

    Wraps a NumPy or PyTorch matrix into a callable that maps a batch of
    signals to measurements, supporting ``torch.autograd`` for gradient-based
    inverse solvers (DPS, PGDM).

    Args:
        A: Measurement matrix of shape ``(M, D)``.  Converted to a
            ``torch.Tensor`` if given as NumPy.

    Returns:
        A callable ``f(x) -> y_pred`` operating on batched tensors of
        shape ``(B, D)`` and returning ``(B, M)``.
    """
    if isinstance(A, np.ndarray):
        A_t = torch.as_tensor(A, dtype=torch.float32)
    else:
        A_t = A.float()

    def _forward(x: torch.Tensor) -> torch.Tensor:
        return x @ A_t.to(x.device).T

    return _forward


# ---------------------------------------------------------------------------
# Classical solvers
# ---------------------------------------------------------------------------

def tikhonov_solve(
    A: np.ndarray,
    y: np.ndarray,
    lam: float = 0.1,
) -> InverseResult:
    """Tikhonov-regularized least squares.

    Solves ``min ||Ax - y||^2 + lam ||x||^2``.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        lam: Regularisation strength.

    Returns:
        Reconstruction result.
    """
    ata = A.T @ A
    rhs = A.T @ y
    x_hat = np.linalg.solve(ata + lam * np.eye(A.shape[1]), rhs)
    return InverseResult(x_hat=x_hat, method="tikhonov", metadata={"lambda": lam})


def lasso_solve(
    A: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
) -> InverseResult:
    """LASSO sparse recovery via coordinate descent.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        alpha: L1-regularisation coefficient.

    Returns:
        Reconstruction result.
    """
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
    model.fit(A, y)
    return InverseResult(x_hat=model.coef_, method="lasso", metadata={"alpha": alpha})


def least_squares_solve(A: np.ndarray, y: np.ndarray) -> InverseResult:
    """Unregularized minimum-norm least squares.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.

    Returns:
        Reconstruction result.
    """
    x_hat, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return InverseResult(x_hat=x_hat, method="least_squares", metadata={})


# ---------------------------------------------------------------------------
# Vanilla diffusion posterior selection
# ---------------------------------------------------------------------------

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
    measurement residual ``||Ax - y||`` (approximate posterior mode).

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        train_signals: Training corpus ``(N, D)`` for the DDPM prior.
        n_samples: Number of candidate samples to generate.
        guidance_scale: Exponential weighting temperature.
        timesteps: Diffusion schedule length.
        epochs: DDPM training epochs.
        seed: Random seed.
        device: Compute device.

    Returns:
        Weighted-average reconstruction result.
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


# ---------------------------------------------------------------------------
# DPS  (Chung et al., "Diffusion Posterior Sampling", 2023)
# ---------------------------------------------------------------------------

def DPS_solve(
    A: np.ndarray,
    y: np.ndarray,
    ddpm: DDPMResult,
    zeta: float = 1.0,
    seed: int = 42,
) -> InverseResult:
    """Diffusion Posterior Sampling (DPS).

    At each reverse step the standard DDPM mean is corrected by a
    measurement-consistency gradient:

    .. math::
        x_{t-1} = \\text{DDPM\\_reverse}(x_t)
                 - \\zeta \\nabla_{x_t} \\|y - A\\,\\hat x_0(x_t)\\|^2

    where ``hat x_0`` is the one-step Tweedie denoised estimate.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        ddpm: Pre-trained ``DDPMResult``.
        zeta: Step size for the measurement gradient.
        seed: Random seed.

    Returns:
        Reconstruction result with method ``"dps"``.
    """
    set_torch_seed(seed)
    dev = torch.device(ddpm.device)
    model = ddpm.model.to(dev)
    model.eval()
    schedule = ddpm.schedule

    fwd = _measurement_fn(A)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=dev)

    dim = model.net[0].in_features - 1
    x = torch.randn(1, dim, device=dev)
    T = len(schedule.betas)

    for t in reversed(range(T)):
        x_in = x.detach().requires_grad_(True)
        t_batch = torch.full((1,), t, device=dev, dtype=torch.long)
        eps_pred = model(x_in, t_batch)

        alpha_bar_t = float(schedule.alpha_bars[t])
        x0_hat = (x_in - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / np.sqrt(
            alpha_bar_t
        )

        residual = y_t - fwd(x0_hat).squeeze(0)
        loss = (residual ** 2).sum()
        loss.backward()

        grad = x_in.grad.detach()

        with torch.no_grad():
            beta = schedule.betas[t]
            alpha = schedule.alphas[t]
            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_pred.detach())
            mean = mean - zeta * grad

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + np.sqrt(beta) * noise
            else:
                x = mean

    return InverseResult(
        x_hat=x.detach().cpu().numpy().squeeze(0),
        method="dps",
        metadata={"zeta": zeta},
    )


# ---------------------------------------------------------------------------
# PGDM  (Song et al., "Pseudoinverse-Guided Diffusion Models", 2023)
# ---------------------------------------------------------------------------

def PGDM_solve(
    A: np.ndarray,
    y: np.ndarray,
    ddpm: DDPMResult,
    seed: int = 42,
) -> InverseResult:
    """Pseudoinverse-Guided Diffusion Model (PGDM).

    Replaces the DPS gradient step with an analytic pseudoinverse
    correction, projecting the Tweedie estimate onto the measurement
    subspace at every reverse step:

    .. math::
        \\hat x_0^+ = \\hat x_0 + A^+ (y - A \\hat x_0)

    The corrected ``hat x_0^+`` is then used in the standard DDPM
    posterior mean formula.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        ddpm: Pre-trained ``DDPMResult``.
        seed: Random seed.

    Returns:
        Reconstruction result with method ``"pgdm"``.
    """
    set_torch_seed(seed)
    dev = torch.device(ddpm.device)
    model = ddpm.model.to(dev)
    model.eval()
    schedule = ddpm.schedule

    A_t = torch.as_tensor(A, dtype=torch.float32, device=dev)
    A_pinv = torch.linalg.pinv(A_t)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=dev)

    dim = model.net[0].in_features - 1
    x = torch.randn(1, dim, device=dev)
    T = len(schedule.betas)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((1,), t, device=dev, dtype=torch.long)
            eps_pred = model(x, t_batch)

            alpha_bar_t = float(schedule.alpha_bars[t])
            x0_hat = (x - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / np.sqrt(
                alpha_bar_t
            )

            measurement_residual = y_t - (x0_hat @ A_t.T).squeeze(0)
            x0_corrected = x0_hat + (measurement_residual @ A_pinv.T)

            beta = schedule.betas[t]
            alpha = schedule.alphas[t]
            alpha_bar_prev = float(schedule.alpha_bars[t - 1]) if t > 0 else 1.0

            posterior_mean_coef_x0 = (
                np.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar_t)
            )
            posterior_mean_coef_xt = (
                np.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            )
            mean = posterior_mean_coef_x0 * x0_corrected + posterior_mean_coef_xt * x

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + np.sqrt(beta) * noise
            else:
                x = mean

    return InverseResult(
        x_hat=x.cpu().numpy().squeeze(0),
        method="pgdm",
        metadata={},
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
    **kwargs: object,
) -> InverseResult:
    """Dispatch to a classical inverse problem solver.

    Args:
        A: Measurement matrix ``(M, D)``.
        y: Observation vector ``(M,)``.
        method: One of ``"least_squares"``, ``"tikhonov"``, ``"lasso"``.
        **kwargs: Forwarded to the chosen solver.

    Returns:
        Reconstruction result.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method not in SOLVERS:
        raise ValueError(f"Unknown solver: {method}. Choose from {list(SOLVERS)}")
    return SOLVERS[method](A, y, **kwargs)
