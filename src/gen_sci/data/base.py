"""Dataset protocols for generative modeling benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GenerationDataset:
    """Samples from a known generative distribution."""

    samples: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def dim(self) -> int:
        return int(self.samples.shape[1]) if self.samples.ndim > 1 else 1


@dataclass
class InverseDataset:
    """Linear inverse problem y = A x + noise with known x."""

    y: np.ndarray
    A: np.ndarray
    x_true: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_measurements(self) -> int:
        return int(self.y.shape[0])

    @property
    def signal_dim(self) -> int:
        return int(self.x_true.shape[0])


@dataclass
class MolecularDataset:
    """Latent molecular representations with property oracle."""

    latents: np.ndarray
    properties: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_molecules(self) -> int:
        return int(self.latents.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1]) if self.latents.ndim > 1 else 1
