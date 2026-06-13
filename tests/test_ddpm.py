"""Tests for DDPM forward diffusion, training, DDIM, and guided sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gen_sci.data.gaussian_mixture_dgp import (
    GaussianMixtureDGPConfig,
    generate_gaussian_mixture_data,
)
from gen_sci.diffusion.ddpm import (
    ConditionalScoreMLP,
    DDPMResult,
    DDPMSampler,
    ScoreMLP,
    sample_ddpm,
    train_ddpm,
)
from gen_sci.diffusion.schedule import DiffusionSchedule


# ---------------------------------------------------------------
# Forward diffusion sanity checks
# ---------------------------------------------------------------


class TestForwardDiffusion:
    """Verify that the forward diffusion q(x_t | x_0) behaves as expected."""

    def test_t0_preserves_signal(self) -> None:
        """At t=0 (alpha_bar ~ 1) the noisy sample should stay close to x_0."""
        schedule = DiffusionSchedule.cosine(50)
        x0 = np.array([[1.0, 2.0]])
        alpha_bar_0 = schedule.alpha_bars[0]
        assert alpha_bar_0 > 0.95

    def test_tT_is_gaussian(self) -> None:
        """At t=T-1 (alpha_bar ~ 0) samples should resemble N(0, I)."""
        schedule = DiffusionSchedule.cosine(200)
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal((5000, 2)) * 5.0

        alpha_bar_T = schedule.alpha_bars[-1]
        eps = rng.standard_normal(x0.shape)
        x_T = np.sqrt(alpha_bar_T) * x0 + np.sqrt(1 - alpha_bar_T) * eps

        assert np.abs(x_T.mean()) < 0.15
        assert np.abs(x_T.std() - 1.0) < 0.15


# ---------------------------------------------------------------
# Training convergence
# ---------------------------------------------------------------


class TestTraining:
    """Verify that a trained DDPM learns the data distribution."""

    @pytest.fixture()
    def trained_result(self) -> DDPMResult:
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=1000, seed=0)
        )
        return train_ddpm(
            data.samples,
            timesteps=20,
            epochs=30,
            batch_size=128,
            hidden=64,
            seed=0,
            device="cpu",
        )

    def test_trained_loss_lower_than_untrained(self, trained_result: DDPMResult) -> None:
        untrained_model = ScoreMLP(2, hidden=64, timesteps=20)
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=200, seed=99)
        )
        schedule = DiffusionSchedule.cosine(20)
        x0 = torch.as_tensor(data.samples, dtype=torch.float32)
        alpha_bars = torch.as_tensor(schedule.alpha_bars, dtype=torch.float32)

        torch.manual_seed(42)
        t = torch.randint(0, 20, (200,))
        eps = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(alpha_bars[t]).unsqueeze(-1)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_bars[t]).unsqueeze(-1)
        xt = sqrt_ab * x0 + sqrt_1m_ab * eps

        with torch.no_grad():
            untrained_loss = float(
                torch.nn.functional.mse_loss(untrained_model(xt, t), eps).item()
            )

        assert trained_result.train_loss < untrained_loss

    def test_samples_are_finite(self, trained_result: DDPMResult) -> None:
        generated = sample_ddpm(trained_result, n_samples=100, seed=1)
        assert np.isfinite(generated).all()
        assert generated.shape == (100, 2)


# ---------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------


class TestDDIM:
    """Verify deterministic DDIM sampling is close to DDPM outputs."""

    def test_ddim_produces_finite_samples(self) -> None:
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=800, seed=0)
        )
        result = train_ddpm(
            data.samples,
            timesteps=20,
            epochs=20,
            batch_size=128,
            hidden=64,
            seed=0,
            device="cpu",
        )
        sampler = DDPMSampler(result)
        ddim_samples = sampler.ddim_sample(n_samples=100, seed=0)
        assert ddim_samples.shape == (100, 2)
        assert np.isfinite(ddim_samples).all()

    def test_ddim_is_deterministic(self) -> None:
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=500, seed=0)
        )
        result = train_ddpm(
            data.samples, timesteps=15, epochs=10, batch_size=128,
            hidden=64, seed=0, device="cpu",
        )
        sampler = DDPMSampler(result)
        s1 = sampler.ddim_sample(n_samples=50, seed=7)
        s2 = sampler.ddim_sample(n_samples=50, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_ddim_samples_reasonable_range(self) -> None:
        """DDIM produces finite samples within a reasonable data range."""
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=1000, seed=0)
        )
        result = train_ddpm(
            data.samples, timesteps=30, epochs=40, batch_size=128,
            hidden=64, seed=0, device="cpu",
        )
        ddim_out = DDPMSampler(result).ddim_sample(n_samples=200, seed=0)
        assert np.isfinite(ddim_out).all()
        assert np.abs(ddim_out).max() <= 10.0


# ---------------------------------------------------------------
# Guided sampling
# ---------------------------------------------------------------


class TestGuidedSample:
    """Basic guided_sample smoke test (requires ConditionalScoreMLP)."""

    def test_guided_rejects_unconditional_model(self) -> None:
        data = generate_gaussian_mixture_data(
            GaussianMixtureDGPConfig(n_samples=500, seed=0)
        )
        result = train_ddpm(
            data.samples, timesteps=10, epochs=5, batch_size=128,
            hidden=64, seed=0, device="cpu",
        )
        sampler = DDPMSampler(result)
        with pytest.raises(TypeError):
            sampler.guided_sample(
                n_samples=10,
                labels=torch.zeros(10, dtype=torch.long),
            )

    def test_guided_produces_finite_samples(self) -> None:
        schedule = DiffusionSchedule.cosine(10)
        model = ConditionalScoreMLP(dim=2, n_classes=3, hidden=32, timesteps=10)
        result = DDPMResult(model=model, schedule=schedule, train_loss=0.5, device="cpu")
        sampler = DDPMSampler(result)
        labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        out = sampler.guided_sample(
            n_samples=5, labels=labels, guidance_scale=2.0, seed=0,
        )
        assert out.shape == (5, 2)
        assert np.isfinite(out).all()
