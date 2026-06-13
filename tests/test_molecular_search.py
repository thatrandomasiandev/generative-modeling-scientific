"""Tests for molecular design search."""

import numpy as np

from gen_sci.data.molecular_dgp import MolecularDGPConfig, generate_molecular_data
from gen_sci.molecular.metrics import regret
from gen_sci.molecular.search import hill_climb_search, random_search


def test_hill_climb_beats_random():
    cfg = MolecularDGPConfig(seed=0)
    data = generate_molecular_data(cfg)
    gt = data.ground_truth
    budget = 100
    random_res = random_search(
        cfg.latent_dim, budget, gt["optimum"], gt["active_dims"], cfg.interaction_strength, seed=0
    )
    climb_res = hill_climb_search(
        cfg.latent_dim, budget, gt["optimum"], gt["active_dims"], cfg.interaction_strength, seed=0
    )
    assert climb_res.best_property >= random_res.best_property - 0.5


def test_regret_non_negative():
    assert regret(1.0, 2.0) == 1.0
    assert regret(3.0, 2.0) == 0.0


def test_search_eval_budget():
    cfg = MolecularDGPConfig(seed=1)
    data = generate_molecular_data(cfg)
    gt = data.ground_truth
    result = random_search(
        cfg.latent_dim, 50, gt["optimum"], gt["active_dims"], cfg.interaction_strength, seed=1
    )
    assert result.n_evaluations == 50
    assert len(result.all_properties) == 50
