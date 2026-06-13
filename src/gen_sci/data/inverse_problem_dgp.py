"""Compressed sensing inverse problem with known sparse ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gen_sci.data.base import GenerationDataset, InverseDataset
from gen_sci.utils.seed import set_seed


@dataclass
class InverseProblemDGPConfig:
    """Configuration for y = A x + noise with sparse x."""

    signal_dim: int = 64
    n_measurements: int = 32
    sparsity: int = 5
    noise_std: float = 0.05
    n_train_signals: int = 2000
    seed: int = 42


def _make_sensing_matrix(
    n_measurements: int, signal_dim: int, rng: np.random.Generator
) -> np.ndarray:
    A = rng.standard_normal((n_measurements, signal_dim))
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    return A.astype(np.float64)


def _sample_sparse_signal(signal_dim: int, sparsity: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(signal_dim, dtype=np.float64)
    support = rng.choice(signal_dim, size=sparsity, replace=False)
    x[support] = rng.standard_normal(sparsity)
    return x


def generate_inverse_problem_data(config: InverseProblemDGPConfig) -> InverseDataset:
    """Generate one inverse problem instance with sparse ground truth."""
    rng = set_seed(config.seed)
    A = _make_sensing_matrix(config.n_measurements, config.signal_dim, rng)
    x_true = _sample_sparse_signal(config.signal_dim, config.sparsity, rng)
    noise = rng.normal(0.0, config.noise_std, size=config.n_measurements)
    y = A @ x_true + noise

    return InverseDataset(
        y=y,
        A=A,
        x_true=x_true,
        metadata={
            "dgp": "compressed_sensing",
            "signal_dim": config.signal_dim,
            "n_measurements": config.n_measurements,
            "sparsity": config.sparsity,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
        ground_truth={
            "support": np.flatnonzero(x_true),
            "noise": noise,
            "condition_number": float(np.linalg.cond(A)),
        },
    )


def generate_signal_corpus(config: InverseProblemDGPConfig) -> GenerationDataset:
    """Generate a corpus of sparse signals for training diffusion priors."""
    rng = set_seed(config.seed + 7)
    signals = np.stack(
        [
            _sample_sparse_signal(config.signal_dim, config.sparsity, rng)
            for _ in range(config.n_train_signals)
        ]
    )
    return GenerationDataset(
        samples=signals,
        metadata={
            "dgp": "sparse_signal_corpus",
            "signal_dim": config.signal_dim,
            "sparsity": config.sparsity,
            "n_train_signals": config.n_train_signals,
            "seed": config.seed,
        },
        ground_truth={"sparsity": config.sparsity},
    )
