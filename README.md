# Generative Modeling for Scientific Applications

A research benchmark suite for **denoising diffusion probabilistic models**, **linear inverse problems**, and **property-guided molecular design** — three generative modeling paradigms central to scientific discovery. All experiments use synthetic DGPs with known data distributions, sparse ground truth, and property oracles, enabling exact evaluation of sample quality, reconstruction accuracy, and search efficiency.

The unifying research question: *how do generative models recover multi-modal distributions, solve underdetermined inverse problems, and guide design in high-dimensional spaces with expensive oracles?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Diffusion** | Learn and sample from multi-modal distributions | DDPM with cosine noise schedule | MMD, mode coverage, train loss |
| **Inverse** | Recover sparse signals from underdetermined measurements | Tikhonov, LASSO, diffusion posterior sampling | Relative L2, support Jaccard, PSNR |
| **Molecular** | Optimize molecular properties in latent space | Random search, hill-climbing, guided diffusion | Regret, normalized score, top-k hit rate |

---

## Module 1: Diffusion models

### Problem formulation

**Denoising diffusion probabilistic models (DDPMs)** (Ho et al., 2020) learn to reverse a gradual noising process. Given data x₀, a forward process adds Gaussian noise over T timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$$

The model learns to predict noise ε_θ(x_t, t) and generates samples by iteratively denoising from x_T ~ N(0, I). Song et al. (2021) unified this under score-based generative modeling via stochastic differential equations.

### Implemented method

- **DDPM** with **cosine noise schedule** (`diffusion/schedule.py`) — smoother signal-to-noise ratio than linear schedule (Nichol & Dhariwal, 2021)
- **ScoreMLP** network for ε-prediction on 2D data (`diffusion/ddpm.py`)

### Synthetic DGP (`data/gaussian_mixture_dgp.py`)

2D Gaussian mixture with configurable **mode separation**. Ground-truth component means and weights enable exact mode coverage measurement.

### Evaluation metrics

- **MMD** (maximum mean discrepancy): Kernel two-sample test between generated and true samples (Gretton et al., 2012)
- **Mode coverage:** Fraction of mixture components with at least one generated sample nearby
- **Train loss:** Denoising objective convergence

---

## Module 2: Inverse problems

### Problem formulation

Given measurements y = Ax + η where A ∈ ℝ^{m×n} with m < n (underdetermined), recover sparse signal x. This is the **compressed sensing** framework (Donoho, 2006; Candès et al., 2006).

### Implemented solvers

| Solver | Approach | Reference |
|--------|----------|-----------|
| **Least squares** | min ‖Ax − y‖² | Baseline (ill-posed) |
| **Tikhonov** | min ‖Ax − y‖² + λ‖x‖² | Tikhonov & Arsenin (1977) |
| **LASSO** | min ‖Ax − y‖² + λ‖x‖₁ | Tibshirani (1996) |
| **Diffusion posterior sampling** | Sample from p(x|y) via diffusion prior + likelihood guidance | Chung et al. (2023) |

LASSO promotes sparsity via ℓ₁ regularization, recovering signals when A satisfies restricted isometry properties (Candès et al., 2006). Diffusion posterior sampling combines a learned generative prior with measurement consistency at each denoising step (Chung et al., 2023; Song et al., 2021).

### Synthetic DGP (`data/inverse_problem_dgp.py`)

- Sparse ground-truth x with known support
- Random measurement matrix A
- Additive Gaussian noise η with tunable level

### Evaluation metrics

- **Relative L2:** ‖x̂ − x‖ / ‖x‖
- **Support Jaccard:** Overlap of estimated and true nonzero indices
- **PSNR:** Peak signal-to-noise ratio of reconstruction
- **Measurement consistency:** ‖Ax̂ − y‖

---

## Module 3: Molecular design

### Problem formulation

Search a high-dimensional **composition space** for molecules maximizing a property f(x), where each evaluation of f is expensive (simulating a DFT calculation or assay). This is **Bayesian optimization** / **molecular design** (Gómez-Bombarelli et al., 2018; Jumper et al., 2021).

### Implemented search strategies

| Strategy | Approach | Reference |
|----------|----------|-----------|
| **Random search** | Uniform sampling in latent space | Baseline |
| **Hill-climbing** | Greedy local improvement | Local search |
| **Guided diffusion** | Sample from diffusion model biased toward high-property regions | Property-guided generation |

### Synthetic DGP (`data/molecular_dgp.py`)

Property oracle f(x) is smooth in a **low-dimensional active subspace** embedded in high-dimensional composition space. Ground-truth optimum is known for regret computation.

### Evaluation metrics

- **Regret:** f(x*) − f(x̂_best)
- **Normalized score:** (f(x̂) − f_random) / (f(x*) − f_random)
- **Top-k hit rate:** Fraction of top-k found within evaluation budget

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/diffusion_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/diffusion_benchmark.yaml --module diffusion
python scripts/run_benchmark.py --config configs/inverse_benchmark.yaml --module inverse
python scripts/run_benchmark.py --config configs/molecular_benchmark.yaml --module molecular

pytest
```

---

## Project layout

```
src/gen_sci/
├── data/          # Gaussian mixture, inverse problem, molecular DGPs
├── diffusion/     # DDPM training, cosine schedule, sampling, MMD metrics
├── inverse/       # Tikhonov, LASSO, diffusion posterior solvers
├── molecular/     # Random, hill-climb, guided diffusion search
└── evaluation/    # Benchmark runner and reporting
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_synthetic_dgp_walkthrough.ipynb` | Validate mixture modes and sparse signals |
| `02_diffusion_benchmark.ipynb` | DDPM training curves and MMD evaluation |
| `03_inverse_problem.ipynb` | Solver comparison across noise levels |
| `04_molecular_design.ipynb` | Search strategy regret curves |

---

## Implementation notes

- DDPM operates on **2D data** for tractable visualization and exact MMD computation
- Diffusion posterior sampling uses a simplified likelihood gradient; full DPS/LGD (Chung et al., 2022) with Tweedie corrections is future work
- Molecular search operates in a **latent space** learned by the diffusion model, not in explicit SMILES/graph space

---

## References

- Candès, E. J., Romberg, J., & Tao, T. (2006). Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. *IEEE TIT*.
- Chung, H., et al. (2022). Improving diffusion models for inverse problems using manifold constraints. *NeurIPS*.
- Chung, H., et al. (2023). Diffusion posterior sampling for general noisy inverse problems. *ICLR*.
- Donoho, D. L. (2006). Compressed sensing. *IEEE TIT*.
- Gómez-Bombarelli, R., et al. (2018). Automatic chemical design using a data-driven continuous representation of molecules. *ACS Central Science*.
- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
- Song, Y., et al. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*.
- Tikhonov, A. N., & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems*. Winston.

---

## Future work

- Score-based posterior sampling with DPS/LGD gradient guidance
- Equivariant 3D molecular diffusion (Hoogeboom et al., 2022)
- Real inverse problems: MRI k-space, CT sinograms
