"""Benchmark runner for diffusion, inverse, and molecular modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gen_sci.data.gaussian_mixture_dgp import GaussianMixtureDGPConfig, generate_gaussian_mixture_data, sample_gaussian_mixture
from gen_sci.data.inverse_problem_dgp import InverseProblemDGPConfig, generate_inverse_problem_data, generate_signal_corpus
from gen_sci.data.molecular_dgp import MolecularDGPConfig, generate_molecular_data
from gen_sci.diffusion.ddpm import sample_ddpm, train_ddpm
from gen_sci.diffusion.metrics import maximum_mean_discrepancy, mode_coverage
from gen_sci.inverse.metrics import measurement_consistency, psnr, relative_l2_error, support_recovery
from gen_sci.inverse.solvers import diffusion_posterior_sample, solve_inverse
from gen_sci.molecular.metrics import normalized_score, regret, top_k_hit_rate
from gen_sci.molecular.search import guided_diffusion_search, hill_climb_search, random_search
from gen_sci.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys if isinstance(results[0][k], (int, float))}


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_diffusion_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Train DDPM on Gaussian mixture and evaluate sample quality."""
    seeds = config.get("seeds", [42])
    separations = config.get("separations", [3.0, 5.0])
    sample_sizes = config.get("train_sizes", [2000, 4000])
    timesteps = config.get("timesteps", 50)
    epochs = config.get("epochs", 80)
    device = config.get("device", "cpu")

    all_results = []
    for separation in separations:
        for n_samples in sample_sizes:
            seed_results = []
            for seed in seeds:
                dgp_cfg = GaussianMixtureDGPConfig(
                    n_samples=n_samples,
                    separation=separation,
                    seed=seed,
                )
                data = generate_gaussian_mixture_data(dgp_cfg)
                ddpm = train_ddpm(
                    data.samples,
                    timesteps=timesteps,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                )
                generated = sample_ddpm(ddpm, n_samples=1000, seed=seed + 1)
                reference = sample_gaussian_mixture(dgp_cfg, n_samples=1000)
                means = data.ground_truth["means"]
                seed_results.append(
                    {
                        "mmd": maximum_mean_discrepancy(generated, reference),
                        "mode_coverage": mode_coverage(generated, means),
                        "train_loss": ddpm.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "separation": separation,
                    "train_size": n_samples,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "diffusion", "results": all_results}


def run_inverse_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare classical and diffusion-guided inverse solvers."""
    seeds = config.get("seeds", [42])
    noise_levels = config.get("noise_levels", [0.02, 0.08])
    solvers = config.get("solvers", ["least_squares", "tikhonov", "lasso"])
    device = config.get("device", "cpu")

    all_results = []
    for noise_std in noise_levels:
        for solver in solvers:
            seed_results = []
            for seed in seeds:
                inv_cfg = InverseProblemDGPConfig(noise_std=noise_std, seed=seed)
                data = generate_inverse_problem_data(inv_cfg)
                if solver == "tikhonov":
                    result = solve_inverse(data.A, data.y, method="tikhonov", lam=0.05)
                elif solver == "lasso":
                    result = solve_inverse(data.A, data.y, method="lasso", alpha=0.005)
                else:
                    result = solve_inverse(data.A, data.y, method=solver)
                seed_results.append(
                    {
                        "rel_l2": relative_l2_error(result.x_hat, data.x_true),
                        "support_jaccard": support_recovery(result.x_hat, data.x_true),
                        "psnr": psnr(result.x_hat, data.x_true),
                        "meas_consistency": measurement_consistency(data.A, result.x_hat, data.y),
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "solver": solver,
                    "noise_std": noise_std,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )

        diff_results = []
        for seed in seeds:
            inv_cfg = InverseProblemDGPConfig(noise_std=noise_std, seed=seed)
            data = generate_inverse_problem_data(inv_cfg)
            corpus = generate_signal_corpus(inv_cfg)
            result = diffusion_posterior_sample(
                data.A,
                data.y,
                corpus.samples,
                n_samples=64,
                seed=seed,
                device=device,
                epochs=config.get("epochs", 50),
                timesteps=config.get("timesteps", 30),
            )
            diff_results.append(
                {
                    "rel_l2": relative_l2_error(result.x_hat, data.x_true),
                    "support_jaccard": support_recovery(result.x_hat, data.x_true),
                    "psnr": psnr(result.x_hat, data.x_true),
                    "meas_consistency": measurement_consistency(data.A, result.x_hat, data.y),
                }
            )
        mean = _aggregate(diff_results)
        std = _aggregate_std(diff_results)
        all_results.append(
            {
                "solver": "diffusion_posterior",
                "noise_std": noise_std,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )

    return {"module": "inverse", "results": all_results}


def run_molecular_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare molecular design search strategies."""
    seeds = config.get("seeds", [42])
    methods = config.get("methods", ["random", "hill_climb", "guided_diffusion"])
    budgets = config.get("eval_budgets", [200, 500])
    device = config.get("device", "cpu")

    all_results = []
    for budget in budgets:
        for method in methods:
            seed_results = []
            for seed in seeds:
                mol_cfg = MolecularDGPConfig(seed=seed)
                data = generate_molecular_data(mol_cfg)
                gt = data.ground_truth
                if method == "random":
                    result = random_search(
                        mol_cfg.latent_dim,
                        budget,
                        gt["optimum"],
                        gt["active_dims"],
                        mol_cfg.interaction_strength,
                        seed=seed,
                    )
                elif method == "hill_climb":
                    result = hill_climb_search(
                        mol_cfg.latent_dim,
                        budget,
                        gt["optimum"],
                        gt["active_dims"],
                        mol_cfg.interaction_strength,
                        seed=seed,
                    )
                else:
                    result = guided_diffusion_search(
                        data.latents,
                        budget,
                        gt["optimum"],
                        gt["active_dims"],
                        mol_cfg.interaction_strength,
                        seed=seed,
                        device=device,
                        epochs=config.get("epochs", 50),
                    )

                random_baseline = float(np.mean(result.all_properties[: max(budget // 10, 5)]))
                seed_results.append(
                    {
                        "best_property": result.best_property,
                        "regret": regret(result.best_property, gt["oracle_value"]),
                        "normalized_score": normalized_score(
                            result.best_property, random_baseline, gt["oracle_value"]
                        ),
                        "top10_hit_rate": top_k_hit_rate(
                            result.all_properties, gt["top_1_percent_threshold"], k=10
                        ),
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "method": method,
                    "eval_budget": budget,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "molecular", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    merged = {**load_config(Path(config_path).parent / "default.yaml"), **config}

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("diffusion", "all"):
        results["modules"]["diffusion"] = run_diffusion_benchmark(merged)
    if module in ("inverse", "all"):
        results["modules"]["inverse"] = run_inverse_benchmark(merged)
    if module in ("molecular", "all"):
        results["modules"]["molecular"] = run_molecular_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from gen_sci.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir
