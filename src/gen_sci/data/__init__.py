"""Synthetic data generators with ground-truth accessors."""

from gen_sci.data.base import GenerationDataset, InverseDataset, MolecularDataset
from gen_sci.data.gaussian_mixture_dgp import (
    GaussianMixtureDGPConfig,
    generate_gaussian_mixture_data,
    sample_gaussian_mixture,
)
from gen_sci.data.inverse_problem_dgp import (
    InverseProblemDGPConfig,
    generate_inverse_problem_data,
    generate_signal_corpus,
)
from gen_sci.data.molecular_dgp import MolecularDGPConfig, generate_molecular_data, property_oracle

__all__ = [
    "GenerationDataset",
    "GaussianMixtureDGPConfig",
    "InverseDataset",
    "InverseProblemDGPConfig",
    "MolecularDGPConfig",
    "MolecularDataset",
    "generate_gaussian_mixture_data",
    "generate_inverse_problem_data",
    "generate_molecular_data",
    "generate_signal_corpus",
    "property_oracle",
    "sample_gaussian_mixture",
]
