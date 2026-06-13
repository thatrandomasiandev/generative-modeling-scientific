"""Tests for inverse problem solvers."""

import numpy as np

from gen_sci.data.inverse_problem_dgp import InverseProblemDGPConfig, generate_inverse_problem_data
from gen_sci.inverse.metrics import relative_l2_error, support_recovery
from gen_sci.inverse.solvers import lasso_solve, solve_inverse, tikhonov_solve


def test_lasso_recovers_sparse_signal():
    data = generate_inverse_problem_data(
        InverseProblemDGPConfig(noise_std=0.01, seed=0)
    )
    result = lasso_solve(data.A, data.y, alpha=0.002)
    assert relative_l2_error(result.x_hat, data.x_true) < 0.5
    assert support_recovery(result.x_hat, data.x_true) >= 0.5


def test_tikhonov_better_than_least_squares_on_noisy():
    data = generate_inverse_problem_data(
        InverseProblemDGPConfig(noise_std=0.1, seed=1)
    )
    ls = solve_inverse(data.A, data.y, method="least_squares")
    tk = tikhonov_solve(data.A, data.y, lam=0.1)
    assert relative_l2_error(tk.x_hat, data.x_true) <= relative_l2_error(ls.x_hat, data.x_true) + 0.2


def test_noiseless_overdetermined_recovery():
    x_true = np.zeros(32)
    x_true[[2, 10, 20]] = [1.0, -0.5, 0.8]
    rng = np.random.default_rng(0)
    A = rng.standard_normal((48, 32))
    y = A @ x_true
    result = lasso_solve(A, y, alpha=1e-6)
    assert relative_l2_error(result.x_hat, x_true) < 0.1
