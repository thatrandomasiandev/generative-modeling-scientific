"""Tests for DDPM training and sampling."""

import numpy as np

from gen_sci.data.gaussian_mixture_dgp import GaussianMixtureDGPConfig, generate_gaussian_mixture_data
from gen_sci.diffusion.ddpm import sample_ddpm, train_ddpm
from gen_sci.diffusion.metrics import maximum_mean_discrepancy, mode_coverage


def test_ddpm_trains_and_samples():
    data = generate_gaussian_mixture_data(GaussianMixtureDGPConfig(n_samples=800, seed=0))
    result = train_ddpm(data.samples, timesteps=20, epochs=15, batch_size=128, seed=0, device="cpu")
    assert result.train_loss < 1.0
    generated = sample_ddpm(result, n_samples=200, seed=1)
    assert generated.shape == (200, 2)
    assert np.isfinite(generated).all()


def test_mmd_decreases_with_better_samples():
    ref = np.random.default_rng(0).standard_normal((100, 2))
    good = ref + 0.05 * np.random.default_rng(1).standard_normal(ref.shape)
    bad = np.random.default_rng(2).standard_normal((100, 2)) * 5
    assert maximum_mean_discrepancy(good, ref) < maximum_mean_discrepancy(bad, ref)


def test_mode_coverage_perfect():
    means = np.array([[0, 0], [5, 0], [0, 5]], dtype=float)
    generated = means + 0.01 * np.random.default_rng(0).standard_normal((3, 2))
    assert mode_coverage(generated, means, threshold=0.5) == 1.0
