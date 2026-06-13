"""Tests for the flow matching module."""

from __future__ import annotations

import numpy as np
import torch

from gen_sci.flow.flow_matching import (
    FlowMatchingModel,
    conditional_flow_target,
    sample_flow_matching,
    train_flow_matching,
)


# ---------------------------------------------------------------
# Vector field direction
# ---------------------------------------------------------------


class TestConditionalFlowTarget:
    """Verify the analytic conditional velocity u_t(x | x_1)."""

    def test_direction_points_toward_target(self) -> None:
        """The target velocity should point from x_0 toward x_1."""
        x0 = torch.zeros(10, 2)
        x1 = torch.ones(10, 2) * 3.0
        t = torch.full((10, 1), 0.5)

        u = conditional_flow_target(x0, x1, t)
        for i in range(10):
            direction = x1[i] - x0[i]
            cos_sim = float(
                torch.dot(u[i], direction)
                / (torch.norm(u[i]) * torch.norm(direction) + 1e-8)
            )
            assert cos_sim > 0.99

    def test_constant_velocity_on_ot_path(self) -> None:
        """On the OT interpolation path the velocity is constant x_1 - x_0."""
        x0 = torch.zeros(1, 2)
        x1 = torch.ones(1, 2) * 3.0
        u_early = conditional_flow_target(x0, x1, torch.tensor([0.1]))
        u_late = conditional_flow_target(x0, x1, torch.tensor([0.9]))
        torch.testing.assert_close(u_early, u_late)
        torch.testing.assert_close(u_early, x1 - x0)


# ---------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------


class TestFlowMatchingModel:
    """Smoke tests for the MLP vector field."""

    def test_output_shape(self) -> None:
        model = FlowMatchingModel(dim=3, hidden=32, n_layers=2)
        x = torch.randn(8, 3)
        t = torch.rand(8, 1)
        out = model(x, t)
        assert out.shape == (8, 3)

    def test_deterministic(self) -> None:
        model = FlowMatchingModel(dim=2, hidden=16, n_layers=2)
        model.eval()
        x = torch.randn(4, 2)
        t = torch.rand(4, 1)
        with torch.no_grad():
            y1 = model(x, t)
            y2 = model(x, t)
        torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------
# Training and sampling
# ---------------------------------------------------------------


class TestTrainAndSample:
    """End-to-end training and generation for flow matching."""

    def test_training_loss_decreases(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.standard_normal((500, 2)) + np.array([3.0, -3.0])

        result = train_flow_matching(
            target,
            epochs=60,
            batch_size=128,
            hidden=64,
            n_layers=2,
            seed=0,
            device="cpu",
        )
        assert result.train_loss < 5.0

    def test_samples_map_toward_target(self) -> None:
        """A trained model should shift Gaussian noise toward the data mean."""
        rng = np.random.default_rng(0)
        mean = np.array([4.0, -4.0])
        target = rng.standard_normal((800, 2)) * 0.5 + mean

        result = train_flow_matching(
            target,
            epochs=80,
            batch_size=128,
            hidden=64,
            n_layers=3,
            seed=0,
            device="cpu",
        )
        generated = sample_flow_matching(result, n_samples=300, n_steps=50, seed=1)
        assert np.isfinite(generated).all()

        gen_mean = generated.mean(axis=0)
        assert np.linalg.norm(gen_mean - mean) < 2.0, (
            f"Generated mean {gen_mean} too far from target mean {mean}"
        )

    def test_samples_are_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.standard_normal((200, 2))
        result = train_flow_matching(
            target, epochs=5, batch_size=64, hidden=32, seed=0, device="cpu",
        )
        s1 = sample_flow_matching(result, n_samples=20, n_steps=10, seed=42)
        s2 = sample_flow_matching(result, n_samples=20, n_steps=10, seed=42)
        np.testing.assert_array_equal(s1, s2)
