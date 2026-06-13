# Generative Modeling for Scientific Applications

PhD-level generative modeling suite covering **diffusion models**, **inverse problems**, and **molecular design** — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Diffusion** | DDPM on 2D Gaussian mixtures with cosine schedule | MMD, mode coverage, train loss |
| **Inverse** | Compressed sensing: Tikhonov, LASSO, diffusion posterior | Relative L2, support Jaccard, PSNR |
| **Molecular** | Latent property landscape search | Regret, normalized score, top-k hit rate |

## Assumptions

- **Diffusion:** Data lies on a low-dimensional manifold; score network is expressive enough for the mixture
- **Inverse:** Linear forward model, sparse ground truth, limited measurements (m < n)
- **Molecular:** Property oracle is smooth in a low-dimensional active subspace; latent diffusion captures the training distribution

## Setup

```bash
cd 05-generative-modeling-scientific
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/diffusion_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/diffusion_benchmark.yaml --module diffusion
python scripts/run_benchmark.py --config configs/inverse_benchmark.yaml --module inverse
python scripts/run_benchmark.py --config configs/molecular_benchmark.yaml --module molecular
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/gen_sci/
├── data/          # Synthetic DGPs with ground-truth accessors
├── diffusion/     # DDPM training, sampling, metrics
├── inverse/       # Classical and diffusion-guided solvers
├── molecular/     # Property-guided search algorithms
└── evaluation/    # Benchmark runner and reporting
```

## Notebooks

- `notebooks/01_synthetic_dgp_walkthrough.ipynb` — validate DGPs
- `notebooks/02_diffusion_benchmark.ipynb` — DDPM training and MMD evaluation
- `notebooks/03_inverse_problem.ipynb` — solver comparison on compressed sensing
- `notebooks/04_molecular_design.ipynb` — search strategy comparison

## Future work

- Score-based posterior sampling with gradient guidance (DPS / LGD)
- 3D molecular conformers with equivariant diffusion
- Real inverse problems (MRI k-space, CT sinograms)
- Property-conditioned generation for multi-objective design
