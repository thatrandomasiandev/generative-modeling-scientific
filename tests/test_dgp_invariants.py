"""Tests for synthetic DGP invariants."""

import numpy as np

from gen_sci.data.gaussian_mixture_dgp import GaussianMixtureDGPConfig, generate_gaussian_mixture_data
from gen_sci.data.inverse_problem_dgp import InverseProblemDGPConfig, generate_inverse_problem_data
from gen_sci.data.molecular_dgp import MolecularDGPConfig, generate_molecular_data, property_oracle


def test_gaussian_mixture_shapes():
    data = generate_gaussian_mixture_data(GaussianMixtureDGPConfig(n_samples=500, seed=0))
    assert data.samples.shape == (500, 2)
    assert data.ground_truth["means"].shape == (3, 2)
    assert np.isclose(data.ground_truth["weights"].sum(), 1.0)


def test_inverse_problem_consistency():
    data = generate_inverse_problem_data(InverseProblemDGPConfig(seed=1))
    residual = data.y - data.A @ data.x_true
    np.testing.assert_allclose(residual, data.ground_truth["noise"], rtol=1e-10)
    assert len(data.ground_truth["support"]) == 5


def test_molecular_oracle_optimum_is_peak():
    data = generate_molecular_data(MolecularDGPConfig(n_molecules=1000, seed=2))
    gt = data.ground_truth
    oracle = property_oracle(
        gt["optimum"][None],
        gt["optimum"],
        gt["active_dims"],
        data.metadata["interaction_strength"],
    )[0]
    assert oracle >= np.max(data.properties) - 1e-6
    assert gt["oracle_value"] >= np.quantile(data.properties, 0.99)


def test_molecular_latent_dim():
    cfg = MolecularDGPConfig(latent_dim=12, seed=3)
    data = generate_molecular_data(cfg)
    assert data.latents.shape == (cfg.n_molecules, 12)
