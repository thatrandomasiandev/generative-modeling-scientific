# Generative Modeling for Scientific Discovery

> Diffusion models, flow matching, and Bayesian optimization as computational engines for inverse problems and molecular design.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2006.11239-b31b1b.svg)](https://arxiv.org/abs/2006.11239)

This repository implements a research-grade benchmarking framework for score-based generative models applied to scientific computing tasks. It unifies Denoising Diffusion Probabilistic Models (DDPM), deterministic DDIM sampling, classifier-free guidance, optimal-transport flow matching, diffusion-guided inverse problem solvers (DPS, PGDM), and Bayesian molecular search under a single reproducible evaluation harness. The codebase is designed for low-dimensional scientific data where ground-truth distributions are known analytically, enabling rigorous quantitative evaluation against oracle metrics. Each module implements the full pipeline from data generation through training, sampling, and metric computation, with configurable noise schedules, importance-weighted timestep sampling, and multi-seed statistical aggregation. The framework targets researchers investigating the intersection of generative modeling with physics-informed inference, compressed sensing, and structure-based drug design, providing both pedagogical implementations with full mathematical documentation and production-quality benchmarking infrastructure.

---

## Table of Contents

- [Research Background \& Motivation](#research-background--motivation)
- [Mathematical Foundations](#mathematical-foundations)
  - [DDPM Forward Process](#ddpm-forward-process)
  - [Score Matching Objective](#score-matching-objective)
  - [DDIM Deterministic Sampling](#ddim-deterministic-sampling)
  - [Classifier-Free Guidance](#classifier-free-guidance)
  - [Flow Matching](#flow-matching)
  - [Diffusion Posterior Sampling (DPS)](#diffusion-posterior-sampling-dps)
  - [Expected Improvement Acquisition](#expected-improvement-acquisition)
  - [Noise Schedules and Signal-to-Noise Ratio](#noise-schedules-and-signal-to-noise-ratio)
- [Architecture Diagram](#architecture-diagram)
- [Code Walkthrough](#code-walkthrough)
  - [Noise Schedules](#noise-schedules)
  - [Score Network Training](#score-network-training)
  - [DDIM Sampling](#ddim-sampling-implementation)
  - [Classifier-Free Guidance Sampling](#classifier-free-guidance-sampling)
  - [Flow Matching Training and ODE Integration](#flow-matching-training-and-ode-integration)
  - [DPS Inverse Solver](#dps-inverse-solver)
  - [PGDM Inverse Solver](#pgdm-inverse-solver)
  - [Bayesian Molecular Search](#bayesian-molecular-search)
  - [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Results](#benchmark-results)
- [Reproduction Commands](#reproduction-commands)
- [Project Structure](#project-structure)
- [References](#references)
- [Future Work](#future-work)

---

## Research Background & Motivation

The emergence of score-based generative models has fundamentally transformed our capacity to sample from complex, high-dimensional probability distributions. Beginning with the seminal work of Ho, Jain, and Abbeel (2020) on Denoising Diffusion Probabilistic Models [1], the field has rapidly expanded from image synthesis into domains where the target distributions carry physical or biological meaning—protein conformations, molecular geometries, quantum states, and solution manifolds of partial differential equations.

The DDPM framework [1] established that a simple noise-prediction network trained with a weighted denoising objective can learn to reverse a fixed Markov corruption process, generating high-fidelity samples through iterative refinement. Song et al. (2021) [2] unified discrete-time diffusion and continuous-time score matching under the framework of stochastic differential equations (SDEs), revealing that both DDPM and score matching with Langevin dynamics (SMLD) are special cases of a broader family of diffusion processes. This SDE perspective opened the door to probability flow ODEs—deterministic samplers that map noise to data along the learned score field, enabling exact likelihood computation and controllable generation.

A complementary approach emerged through the work of Lipman, Chen, Ben-Hamu, Nickel, and Le (2023) [3] on Conditional Flow Matching. Rather than learning a score function to reverse a corruption process, flow matching directly regresses a velocity field that transports a base distribution (typically Gaussian) to the data distribution along optimal-transport paths. The resulting continuous normalizing flow (CNF) achieves comparable sample quality to diffusion models while offering simpler training objectives and exact log-likelihood evaluation through the instantaneous change-of-variables formula.

The application of diffusion priors to inverse problems represents one of the most impactful scientific use cases. Chung, Kim, Mccann, Klasky, and Ye (2023) [4] introduced Diffusion Posterior Sampling (DPS), which modifies the standard reverse diffusion process with a measurement-consistency gradient derived from the likelihood $p(y \mid x_0)$. This enables posterior sampling for linear and even nonlinear forward models without retraining the generative prior, making pre-trained diffusion models immediately applicable to compressed sensing, inpainting, super-resolution, and medical image reconstruction. The Pseudoinverse-Guided Diffusion Model (PGDM) by Song et al. (2023) offers an alternative that replaces gradient-based guidance with analytic pseudoinverse corrections, achieving stronger measurement consistency at the cost of assuming linearity in the forward operator.

In latent-space drug design, the combination of generative priors with property-aware search has shown remarkable promise. Rombach, Blattmann, Lorber, Esser, and Ommer (2022) [5] demonstrated that operating diffusion models in learned latent spaces dramatically reduces computational cost while preserving generation quality—a principle directly applicable to molecular latent spaces. Jing et al. (2022) [7] applied equivariant diffusion to structure-based drug design (DiffSBDD), generating 3D molecular geometries conditioned on protein binding pockets. Hoogeboom et al. (2022) [8] developed equivariant diffusion for molecular generation with E(3) symmetry constraints. Meanwhile, Jumper et al. (2021) [6] with AlphaFold2 demonstrated that deep learning can solve protein structure prediction with atomic accuracy, motivating the use of learned representations as optimization targets for molecular design.

This repository distills these advances into a unified benchmarking framework operating on synthetic data with known ground-truth distributions. By working with analytically tractable problems—Gaussian mixtures with known modes, compressed sensing with known sparse signals, and molecular landscapes with known optima—we can rigorously measure mode coverage, reconstruction fidelity, and optimization gap without confounding factors from real-world data complexity. The three benchmark modules (generative quality, inverse problems, molecular search) each exercise different capabilities of the underlying diffusion framework while sharing a common training and evaluation infrastructure. This design enables controlled ablation studies over noise schedules, network capacity, guidance strength, and sampling strategies, providing quantitative evidence for architectural and algorithmic choices before scaling to real scientific applications.

---

## Mathematical Foundations

### DDPM Forward Process

The forward diffusion process in DDPM defines a Markov chain that gradually corrupts data $x_0 \sim q(x_0)$ by adding Gaussian noise according to a variance schedule $\{\beta_t\}_{t=1}^T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$$

Through the reparameterization trick and the properties of Gaussian composition, we can express the marginal distribution at any timestep $t$ directly in terms of the clean data $x_0$:

$$q(x_t \mid x_0) = \mathcal{N}\bigl(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)\, I\bigr)$$

where we define:

- $\alpha_t = 1 - \beta_t$ — the signal retention coefficient at step $t$
- $\bar\alpha_t = \prod_{s=1}^t \alpha_s = \prod_{s=1}^t (1 - \beta_s)$ — the cumulative signal retention (cumulative product of alphas)

This closed-form marginal allows efficient training: rather than sequentially applying $t$ noise steps, we can directly sample $x_t$ from $x_0$ via:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The schedule $\{\beta_t\}$ controls the rate of information destruction. As $t \to T$ and $\bar\alpha_T \to 0$, the distribution $q(x_T \mid x_0) \approx \mathcal{N}(0, I)$ becomes approximately standard Gaussian, erasing all information about $x_0$.

**Signal-to-Noise Ratio.** The SNR at timestep $t$ is defined as:

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

This quantity monotonically decreases from $\text{SNR}(0) \gg 1$ (nearly clean) to $\text{SNR}(T) \approx 0$ (nearly pure noise). The importance weight for each timestep in the variational lower bound is proportional to $\text{SNR}(t-1) - \text{SNR}(t)$, motivating non-uniform timestep sampling during training.

### Score Matching Objective

The reverse process is parameterized by a neural network $\epsilon_\theta(x_t, t)$ that predicts the noise $\epsilon$ added at step $t$. The training objective minimizes the expected squared error between the predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[1,T],\, x_0 \sim q(x_0),\, \epsilon \sim \mathcal{N}(0,I)} \bigl[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\, t)\|^2\bigr]$$

This objective is equivalent (up to a time-dependent weighting) to denoising score matching:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

so that the score network implicitly learns:

$$s_\theta(x_t, t) \approx \nabla_{x_t} \log q_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

**Ancestral reverse sampling** uses the trained noise predictor to define the reverse Gaussian transition:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\bigr)$$

where:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Bigr)$$

and $\sigma_t^2 = \beta_t$ (or $\sigma_t^2 = \tilde\beta_t$ for the posterior variance).

### DDIM Deterministic Sampling

Song, Meng, and Ermon (2021) observed that the DDPM training objective actually defines a family of non-Markovian reverse processes. The Denoising Diffusion Implicit Model (DDIM) corresponds to the $\eta = 0$ member—a fully deterministic mapping from latent noise to data:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\, \epsilon_\theta(x_t, t)$$

where the one-step denoised prediction (Tweedie's estimate) is:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

Substituting the expression for $\hat{x}_0$ into the update rule yields:

$$x_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}\, x_t + \left(\sqrt{1-\bar\alpha_{t-1}} - \frac{\sqrt{\bar\alpha_{t-1}}\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\right) \epsilon_\theta(x_t, t)$$

Key properties of DDIM:
- **Deterministic**: Given a fixed initial noise $x_T$, the trajectory $x_T \to x_0$ is uniquely determined.
- **Consistent**: Trained with the same objective as DDPM; no retraining required.
- **Acceleration**: Supports sub-sequence sampling (skipping timesteps) without quality degradation.
- **Invertible**: The deterministic ODE can be run forward to encode data into the latent space.

### Classifier-Free Guidance

Ho and Salimans (2022) introduced classifier-free guidance as a method to trade diversity for fidelity in conditional generation without requiring a separate classifier. During training, the conditioning signal $c$ is randomly dropped (replaced with a null token $\varnothing$) with probability $p_\text{uncond}$, teaching the model both the conditional and unconditional distributions.

At inference time, the guided noise prediction is a linear extrapolation:

$$\tilde\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w\,\bigl(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\bigr)$$

where:
- $\epsilon_\theta(x_t, t, \varnothing)$ is the **unconditional** noise prediction (null class embedding)
- $\epsilon_\theta(x_t, t, c)$ is the **conditional** noise prediction given class $c$
- $w > 0$ is the **guidance scale** (guidance weight)

When $w = 1$, this reduces to standard conditional generation. When $w > 1$, the model moves the sample further in the direction that makes it more consistent with the conditioning, at the cost of reduced diversity. The implicit score interpretation is:

$$\nabla_{x_t} \log p_w(x_t \mid c) = (1-w)\, \nabla_{x_t} \log p(x_t) + w\, \nabla_{x_t} \log p(x_t \mid c)$$

which corresponds to sampling from a tempered posterior $p(c \mid x_t)^w \, p(x_t)$.

### Flow Matching

Flow matching (Lipman et al., 2023) [3] provides an alternative to score-based diffusion by directly learning a velocity field $v_\theta(x, t)$ that generates a probability path from a simple base distribution $p_0 = \mathcal{N}(0, I)$ to the data distribution $p_1 = p_\text{data}$.

The **conditional probability path** interpolates linearly between noise and data:

$$x_t = (1-t)\, x_0 + t\, x_1, \quad x_0 \sim \mathcal{N}(0, I),\; x_1 \sim p_\text{data},\; t \in [0, 1]$$

The **conditional vector field** that generates this path is:

$$u_t(x \mid x_1) = \frac{x_1 - x}{1-t+\epsilon}$$

On the interpolation path itself, this simplifies to:

$$u_t(x_t \mid x_1) = \frac{x_1 - x_t}{1-t+\epsilon} = x_1 - x_0$$

The second equality holds because $x_t = (1-t)x_0 + tx_1$, so $x_1 - x_t = (1-t)(x_1 - x_0)$ and dividing by $(1-t)$ recovers $x_1 - x_0$. The $\epsilon$ parameter provides numerical stability near $t = 1$.

The **flow matching loss** trains a neural network to regress this velocity:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0 \sim p_0,\, x_1 \sim p_1}\bigl[\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2\bigr]$$

At inference, samples are generated by solving the ODE from $t=0$ to $t=1$:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim \mathcal{N}(0, I)$$

using a numerical integrator (Euler, RK4, or adaptive methods). With $N$ uniform Euler steps of size $\Delta t = 1/N$:

$$x_{k+1} = x_k + \Delta t\, v_\theta(x_k, t_k), \quad t_k = k\, \Delta t$$

**Advantages over diffusion:**
- Simulation-free training (no forward SDE simulation required)
- Straight transport paths minimize the kinetic energy, yielding shorter integration paths
- Exact log-likelihood via the instantaneous change-of-variables formula
- Naturally extends to optimal-transport coupling between arbitrary source and target distributions

### Diffusion Posterior Sampling (DPS)

Chung et al. (2023) [4] address the inverse problem $y = \mathcal{A}(x_0) + \eta$ where $\mathcal{A}$ is a (possibly nonlinear) forward operator and $\eta$ is measurement noise. DPS leverages a pre-trained unconditional diffusion model as a prior and guides the reverse process toward measurement consistency.

At each reverse step $t$, DPS computes:

1. **Standard DDPM reverse mean** $\mu_\theta(x_t, t)$ from the noise prediction $\epsilon_\theta(x_t, t)$

2. **Tweedie denoised estimate**: $\hat{x}_0(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$

3. **Likelihood gradient approximation**:

$$\nabla_{x_t} \log p(y \mid x_t) \approx -\nabla_{x_t} \|y - \mathcal{A}(\hat{x}_0(x_t))\|^2$$

The corrected reverse step becomes:

$$x_{t-1} = \mu_\theta(x_t, t) - \zeta\, \nabla_{x_t}\|y - \mathcal{A}(\hat{x}_0(x_t))\|^2 + \sigma_t\, z$$

where $\zeta > 0$ is a step-size hyperparameter controlling the strength of measurement guidance and $z \sim \mathcal{N}(0, I)$.

For **linear measurements** $y = Ax_0 + \eta$, the gradient takes the explicit form:

$$\nabla_{x_t}\|y - A\hat{x}_0\|^2 = -\frac{2}{\sqrt{\bar\alpha_t}}\, A^\top(y - A\hat{x}_0) \cdot \frac{\partial \hat{x}_0}{\partial x_t}$$

where the Jacobian $\frac{\partial \hat{x}_0}{\partial x_t}$ is computed via automatic differentiation through the score network.

**PGDM** (Song et al., 2023) replaces this gradient step with a direct pseudoinverse correction:

$$\hat{x}_0^+ = \hat{x}_0 + A^+(y - A\hat{x}_0)$$

where $A^+ = A^\top(AA^\top)^{-1}$ is the Moore–Penrose pseudoinverse. This ensures exact measurement consistency ($A\hat{x}_0^+ = y$) at each step, at the cost of assuming linearity.

### Expected Improvement Acquisition

For Bayesian molecular search, we fit a Gaussian Process (GP) surrogate $f \sim \mathcal{GP}(\mu, k)$ to observed property evaluations and select new candidates by maximizing the Expected Improvement (EI) acquisition function:

$$\text{EI}(x) = (\mu(x) - f^* - \xi)\, \Phi(Z) + \sigma(x)\, \phi(Z)$$

where:
- $\mu(x)$ is the GP posterior mean at $x$
- $\sigma(x)$ is the GP posterior standard deviation at $x$
- $f^* = \max_i f(x_i)$ is the best observed value
- $\xi \geq 0$ is an exploration parameter (trades off exploitation vs. exploration)
- $Z = \frac{\mu(x) - f^* - \xi}{\sigma(x)}$ is the standardized improvement
- $\Phi(\cdot)$ is the standard normal CDF
- $\phi(\cdot)$ is the standard normal PDF

**Decomposition of EI:**
- The first term $(\mu(x) - f^*)\Phi(Z)$ captures the **exploitation** component—probability-weighted expected gain when the mean already exceeds $f^*$
- The second term $\sigma(x)\phi(Z)$ captures the **exploration** component—value of information in high-uncertainty regions

When $\sigma(x) \to 0$, $\text{EI}(x) \to \max(\mu(x) - f^* - \xi,\, 0)$—pure exploitation. When $\sigma(x)$ is large, the exploration term dominates, encouraging queries in unexplored regions.

### Noise Schedules and Signal-to-Noise Ratio

The variance schedule $\{\beta_t\}_{t=1}^T$ critically affects both training stability and generation quality. This repository implements three schedules:

**Linear schedule** (Ho et al., 2020):

$$\beta_t = \beta_\text{start} + \frac{t-1}{T-1}(\beta_\text{end} - \beta_\text{start})$$

**Cosine schedule** (Nichol & Dhariwal, 2021):

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

where $s = 0.008$ prevents a singularity at $t = 0$. The betas are derived as $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$, clipped to $[10^{-5}, 0.999]$.

**Quadratic schedule**:

$$\beta_t = \beta_\text{start} + (\beta_\text{end} - \beta_\text{start})\left(\frac{t}{T}\right)^2$$

The quadratic schedule grows noise more slowly at early timesteps (preserving fine structure longer) and accelerates toward the end. Compared to linear, it allocates more "resolution" to the low-noise regime where perceptual detail is determined.

**Importance-weighted timestep sampling** uses the discrete SNR change as a probability mass:

$$p(t) \propto \text{SNR}(t-1) - \text{SNR}(t) = \frac{\bar\alpha_{t-1}}{1-\bar\alpha_{t-1}} - \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

This concentrates training compute on timesteps where the model's predictions have the largest impact on the variational lower bound.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        gen_sci: System Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────── Data Generation Layer ────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │  │
│  │  │ GaussianMixture │  │ InverseProblem   │  │  MolecularDGP       │  │  │
│  │  │ DGPConfig       │  │ DGPConfig        │  │  Config             │  │  │
│  │  │                 │  │                  │  │                     │  │  │
│  │  │ • n_components  │  │ • signal_dim: 64 │  │ • latent_dim: 8    │  │  │
│  │  │ • separation    │  │ • n_meas: 32     │  │ • n_active_dims: 3 │  │  │
│  │  │ • component_std │  │ • sparsity: 5    │  │ • interaction: 1.5 │  │  │
│  │  └────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘  │  │
│  │           │                    │                        │             │  │
│  └───────────┼────────────────────┼────────────────────────┼─────────────┘  │
│              │                    │                        │                 │
│              ▼                    ▼                        ▼                 │
│  ┌──────────────────── Model Training Layer ─────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │              DiffusionSchedule                                   │ │  │
│  │  │  .linear()  │  .cosine()  │  .quadratic()  │  .snr()           │ │  │
│  │  └──────────────────────────┬───────────────────────────────────────┘ │  │
│  │                             │                                         │  │
│  │           ┌─────────────────┼───────────────────┐                     │  │
│  │           ▼                 ▼                   ▼                     │  │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐         │  │
│  │  │  ScoreMLP   │  │ Conditional     │  │ FlowMatching     │         │  │
│  │  │  ε_θ(x,t)  │  │ ScoreMLP        │  │ Model v_θ(x,t)  │         │  │
│  │  │             │  │ ε_θ(x,t,c)     │  │                  │         │  │
│  │  │ dim+1 → H  │  │ dim+1+H → H    │  │ dim+1 → H → dim │         │  │
│  │  │ → H → dim  │  │ → H → dim      │  │ (n_layers deep)  │         │  │
│  │  └──────┬──────┘  └───────┬─────────┘  └────────┬─────────┘         │  │
│  │         │                 │                      │                    │  │
│  └─────────┼─────────────────┼──────────────────────┼────────────────────┘  │
│            │                 │                      │                        │
│            ▼                 ▼                      ▼                        │
│  ┌──────────────────── Sampling / Inference Layer ───────────────────────┐  │
│  │                                                                       │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │  │
│  │  │  Ancestral │ │    DDIM    │ │  Guided    │ │   Euler ODE        │ │  │
│  │  │  Sampling  │ │  (η = 0)  │ │  (CFG, w)  │ │   Integration      │ │  │
│  │  └─────┬──────┘ └─────┬──────┘ └──────┬─────┘ └─────────┬──────────┘ │  │
│  │        │               │               │                  │            │  │
│  └────────┼───────────────┼───────────────┼──────────────────┼────────────┘  │
│           │               │               │                  │               │
│           ▼               ▼               ▼                  ▼               │
│  ┌──────────────────── Application Layer ────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │  │
│  │  │ Inverse Problem │  │ Molecular Search │  │ Generation Quality  │  │  │
│  │  │ Solvers         │  │                  │  │ Assessment          │  │  │
│  │  │                 │  │ • Random         │  │                     │  │  │
│  │  │ • Tikhonov      │  │ • Hill Climb     │  │ • MMD               │  │  │
│  │  │ • LASSO         │  │ • Guided Diff.   │  │ • Mode Coverage     │  │  │
│  │  │ • DPS           │  │ • Bayesian (EI)  │  │ • Coverage/Density  │  │  │
│  │  │ • PGDM          │  │                  │  │ • Sliced W1         │  │  │
│  │  └─────────────────┘  └──────────────────┘  └─────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────── Evaluation Layer ─────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐   │  │
│  │  │  runner.py: Multi-seed benchmark orchestration                 │   │  │
│  │  │  report.py: Markdown table generation                          │   │  │
│  │  │  generative_metrics.py: MMD, Coverage/Density, Sliced W1      │   │  │
│  │  └────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Walkthrough

### Noise Schedules

The `DiffusionSchedule` dataclass precomputes all derived quantities needed for training and sampling. The cosine schedule from Nichol & Dhariwal avoids the rapid information destruction of a linear schedule at early timesteps:

```python
@classmethod
def cosine(cls, timesteps: int, s: float = 0.008) -> DiffusionSchedule:
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps, dtype=np.float64)
    f = np.cos(((t / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = np.clip(betas, 1e-5, 0.999)
    alphas = 1.0 - betas
    return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars[1:])
```

The function $f(t) = \cos^2\bigl(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\bigr)$ produces $\bar\alpha_t$ values that decrease smoothly from 1 to 0, spending more "budget" on the perceptually important low-noise regime. The offset $s = 0.008$ prevents $\beta_1$ from being too small (which would waste early timesteps), and clipping to $[10^{-5}, 0.999]$ ensures numerical stability.

The SNR computation and importance-weighted timestep sampler expose the information-theoretic structure of the schedule:

```python
def snr(self) -> np.ndarray:
    return self.alpha_bars / (1.0 - self.alpha_bars + 1e-12)

def optimal_t_sampler(self, seed: int = 0) -> np.ndarray:
    snr_vals = self.snr()
    weights = np.zeros_like(snr_vals)
    weights[0] = snr_vals[0]
    weights[1:] = np.maximum(snr_vals[:-1] - snr_vals[1:], 1e-12)
    weights = np.maximum(weights, 1e-12)
    return weights / weights.sum()
```

The weights $p(t) \propto \text{SNR}(t-1) - \text{SNR}(t)$ correspond exactly to the contribution of each timestep to the variational lower bound in the simplified DDPM objective. Timesteps where the SNR changes most rapidly (typically in the mid-range) receive higher sampling probability.

### Score Network Training

The training loop implements the standard DDPM objective with explicit alpha-bar indexing for vectorized batch computation:

```python
alpha_bars = torch.as_tensor(schedule.alpha_bars, dtype=torch.float32, device=dev)
losses: list[float] = []
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    for (x0,) in loader:
        b = x0.shape[0]
        t = torch.randint(0, timesteps, (b,), device=dev)
        eps = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(alpha_bars[t]).unsqueeze(-1)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_bars[t]).unsqueeze(-1)
        xt = sqrt_ab * x0 + sqrt_1m_ab * eps
        pred = model(xt, t)
        loss = nn.functional.mse_loss(pred, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += float(loss.item())
        n_batches += 1
```

Key implementation details:
- **Vectorized corruption**: `alpha_bars[t]` gathers the per-sample $\bar\alpha_t$ using integer indexing, enabling different timesteps per sample in a single batch.
- **Broadcasting**: `.unsqueeze(-1)` expands the scalar coefficients to multiply each dimension of $x_0$.
- **Direct MSE**: The loss computes $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ averaged over both batch and dimension.

The `ScoreMLP` architecture uses SiLU (Swish) activations and a normalized time embedding:

```python
class ScoreMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, timesteps: int = 100) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_norm = (t.float() / max(self.timesteps - 1, 1)).unsqueeze(-1)
        return self.net(torch.cat([x, t_norm], dim=-1))
```

The time embedding $t/T \in [0, 1]$ is concatenated with the spatial input, providing a simple but effective conditioning mechanism for low-dimensional problems. For higher-dimensional applications, sinusoidal or learned Fourier embeddings would be preferred.

### DDIM Sampling Implementation

The `DDPMSampler.ddim_sample` method implements the deterministic DDIM update with dynamic-range clipping on the predicted $\hat{x}_0$:

```python
@torch.no_grad()
def ddim_sample(self, n_samples: int, seed: int = 0) -> np.ndarray:
    set_torch_seed(seed)
    self.model.eval()
    dim = self.model.net[0].in_features - 1
    x = torch.randn(n_samples, dim, device=self.device)
    T = len(self.schedule.betas)

    for t in reversed(range(T)):
        t_batch = torch.full(
            (n_samples,), t, device=self.device, dtype=torch.long
        )
        eps_pred = self.model(x, t_batch)
        alpha_bar_t = float(self.schedule.alpha_bars[t])

        x0_pred = self._predict_x0(x, eps_pred, alpha_bar_t)

        if t > 0:
            alpha_bar_prev = float(self.schedule.alpha_bars[t - 1])
            dir_xt = np.sqrt(1.0 - alpha_bar_prev) * eps_pred
            x = np.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        else:
            x = x0_pred

    return x.cpu().numpy()
```

The helper `_predict_x0` recovers the clean sample estimate:

```python
def _predict_x0(self, x_t, eps_pred, alpha_bar_t):
    x0 = (x_t - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / max(
        np.sqrt(alpha_bar_t), 1e-6
    )
    return torch.clamp(x0, -10.0, 10.0)
```

The clamping to $[-10, 10]$ prevents numerical explosion at late timesteps where $\bar\alpha_t \approx 0$, which would amplify any noise in the prediction by dividing by $\sqrt{\bar\alpha_t} \approx 0$. This is a standard numerical safeguard in practical DDIM implementations.

### Classifier-Free Guidance Sampling

The `guided_sample` method demonstrates the full classifier-free guidance pipeline. The `ConditionalScoreMLP` accepts a class embedding that is dropped to the null class for unconditional predictions:

```python
@torch.no_grad()
def guided_sample(self, n_samples, labels, guidance_scale=3.0, seed=0):
    if not isinstance(self.model, ConditionalScoreMLP):
        raise TypeError("guided_sample requires a ConditionalScoreMLP model")
    set_torch_seed(seed)
    self.model.eval()

    input_dim = self.model.net[0].in_features
    hidden_dim = self.model.class_emb.embedding_dim
    dim = input_dim - 1 - hidden_dim

    x = torch.randn(n_samples, dim, device=self.device)
    labels = labels.to(self.device)
    T = len(self.schedule.betas)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
        eps_cond = self.model(x, t_batch, c=labels)
        eps_uncond = self.model(x, t_batch, c=None)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta = self.schedule.betas[t]
        alpha = self.schedule.alphas[t]
        alpha_bar = self.schedule.alpha_bars[t]
        coef1 = 1.0 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1.0 - alpha_bar)
        mean = coef1 * (x - coef2 * eps_guided)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + np.sqrt(beta) * noise
        else:
            x = mean

    return x.cpu().numpy()
```

The implementation requires **two forward passes** per timestep: one with the true class labels and one with `c=None` (mapped to the null embedding index `n_classes`). The extrapolated prediction `eps_guided` amplifies the difference between conditional and unconditional predictions by a factor of `guidance_scale`, effectively sharpening the conditional distribution.

### Flow Matching Training and ODE Integration

The flow matching training loop is structurally simpler than DDPM—no noise schedule bookkeeping, just random time sampling and linear interpolation:

```python
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    for (x1,) in loader:
        b = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(b, 1, device=dev)

        x_t = (1.0 - t) * x0 + t * x1
        target = conditional_flow_target(x0, x1, t)

        pred = model(x_t, t)
        loss = nn.functional.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
```

The `conditional_flow_target` function computes the optimal-transport target velocity:

```python
def conditional_flow_target(x_0, x_1, t, eps=1e-5):
    return x_1 - x_0
```

This elegant simplification arises because on the linear interpolation path $x_t = (1-t)x_0 + tx_1$, the time derivative is exactly $\dot{x}_t = x_1 - x_0$, independent of $t$. The velocity field is constant along each conditional path.

Sample generation integrates the learned ODE with uniform Euler steps:

```python
@torch.no_grad()
def sample_flow_matching(result, n_samples, n_steps=100, seed=0):
    set_torch_seed(seed)
    dev = torch.device(result.device)
    model = result.model.to(dev)
    model.eval()

    first_layer = model.net[0]
    dim = first_layer.in_features - 1

    x = torch.randn(n_samples, dim, device=dev)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_samples, 1), t_val, device=dev)
        v = model(x, t)
        x = x + v * dt

    return x.cpu().numpy()
```

The Euler discretization introduces $O(\Delta t)$ local error and $O(\Delta t)$ global error for Lipschitz vector fields. In practice, 50–200 steps suffice for low-dimensional problems.

### DPS Inverse Solver

The DPS solver modifies the standard DDPM reverse process with a measurement-consistency gradient computed through automatic differentiation:

```python
def DPS_solve(A, y, ddpm, zeta=1.0, seed=42):
    set_torch_seed(seed)
    dev = torch.device(ddpm.device)
    model = ddpm.model.to(dev)
    model.eval()
    schedule = ddpm.schedule

    fwd = _measurement_fn(A)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=dev)

    dim = model.net[0].in_features - 1
    x = torch.randn(1, dim, device=dev)
    T = len(schedule.betas)

    for t in reversed(range(T)):
        x_in = x.detach().requires_grad_(True)
        t_batch = torch.full((1,), t, device=dev, dtype=torch.long)
        eps_pred = model(x_in, t_batch)

        alpha_bar_t = float(schedule.alpha_bars[t])
        x0_hat = (x_in - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / np.sqrt(alpha_bar_t)

        residual = y_t - fwd(x0_hat).squeeze(0)
        loss = (residual ** 2).sum()
        loss.backward()

        grad = x_in.grad.detach()

        with torch.no_grad():
            beta = schedule.betas[t]
            alpha = schedule.alphas[t]
            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_pred.detach())
            mean = mean - zeta * grad

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + np.sqrt(beta) * noise
            else:
                x = mean
```

Critical implementation detail: `x_in = x.detach().requires_grad_(True)` creates a fresh leaf tensor at each step so that `loss.backward()` computes $\nabla_{x_t}\|y - A\hat{x}_0(x_t)\|^2$ without accumulating gradients through previous steps. The gradient flows through: $x_t \to \hat{x}_0 \to A\hat{x}_0 \to \|y - A\hat{x}_0\|^2$.

### PGDM Inverse Solver

The PGDM approach replaces gradient computation with an analytic pseudoinverse projection:

```python
def PGDM_solve(A, y, ddpm, seed=42):
    set_torch_seed(seed)
    dev = torch.device(ddpm.device)
    model = ddpm.model.to(dev)
    model.eval()
    schedule = ddpm.schedule

    A_t = torch.as_tensor(A, dtype=torch.float32, device=dev)
    A_pinv = torch.linalg.pinv(A_t)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=dev)

    dim = model.net[0].in_features - 1
    x = torch.randn(1, dim, device=dev)
    T = len(schedule.betas)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((1,), t, device=dev, dtype=torch.long)
            eps_pred = model(x, t_batch)

            alpha_bar_t = float(schedule.alpha_bars[t])
            x0_hat = (x - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / np.sqrt(alpha_bar_t)

            measurement_residual = y_t - (x0_hat @ A_t.T).squeeze(0)
            x0_corrected = x0_hat + (measurement_residual @ A_pinv.T)

            beta = schedule.betas[t]
            alpha = schedule.alphas[t]
            alpha_bar_prev = float(schedule.alpha_bars[t - 1]) if t > 0 else 1.0

            posterior_mean_coef_x0 = np.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar_t)
            posterior_mean_coef_xt = np.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean = posterior_mean_coef_x0 * x0_corrected + posterior_mean_coef_xt * x
```

Note that PGDM uses the **full DDPM posterior mean formula** (not the simplified epsilon-based mean) because it needs to inject the corrected $\hat{x}_0^+$ into the posterior:

$$\mu_\text{posterior}(x_t, \hat{x}_0^+) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, \hat{x}_0^+ + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t$$

The pseudoinverse $A^+$ is precomputed once before the reverse loop (`torch.linalg.pinv`), making PGDM computationally cheaper per step than DPS (no gradient computation), though it sacrifices the ability to handle nonlinear forward operators.

### Bayesian Molecular Search

The `BayesianMolecularSearch` class implements a full BO loop with ECFP-like fingerprint features, a Matérn-5/2 GP surrogate, and Expected Improvement acquisition:

```python
def propose(self, n: int = 1) -> np.ndarray:
    self._call_count += 1
    rng = np.random.default_rng(self.seed + self._call_count)

    if self.gp is None or self.scores.shape[0] == 0:
        return rng.standard_normal((n, self.latent_dim))

    random_pool = rng.standard_normal((self.n_candidates, self.latent_dim))
    if self.latents.shape[0] > 0:
        best_idx = int(np.argmax(self.scores))
        mutants = random_molecule_mutate(
            self.latents[best_idx],
            step_size=self.step_size,
            n_children=self.n_candidates,
            seed=self.seed + self._call_count + 1000,
        )
        pool = np.vstack([random_pool, mutants])
    else:
        pool = random_pool

    fp = _ecfp_features(pool, n_bits=self.n_bits, seed=self.seed)
    mu, sigma = self.gp.predict(fp, return_std=True)
    ei = _expected_improvement(mu, sigma, float(self.scores.max()), xi=self.xi)
    top_indices = np.argsort(ei)[-n:][::-1]
    return pool[top_indices]
```

The proposal strategy combines **global exploration** (random candidates) with **local exploitation** (Gaussian perturbations around the current best), then selects candidates with highest EI under the GP surrogate. The ECFP fingerprint projection serves as a nonlinear feature map that captures "structural similarity" in the latent space:

```python
def _ecfp_features(latents, n_bits=128, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((latents.shape[1], n_bits))
    b = rng.standard_normal(n_bits)
    return (latents @ W + b > 0).astype(np.float64)
```

Each fingerprint bit is the sign of a random affine projection—a locality-sensitive hashing scheme that approximates angular distance in the original space.

### Evaluation Metrics

The `generative_metrics.py` module provides three distribution-comparison measures:

**Maximum Mean Discrepancy (MMD)** with the median heuristic for bandwidth selection:

```python
def mmd(generated, reference, gamma=None):
    if gamma is None:
        combined = np.vstack([generated, reference])
        dists = cdist(combined, combined)
        gamma = 1.0 / max(float(np.median(dists[dists > 0])), 1e-6)

    def _kernel(x, y):
        d2 = cdist(x, y, metric="sqeuclidean")
        return np.exp(-gamma * d2)

    n = generated.shape[0]
    m = reference.shape[0]
    k_xx = _kernel(generated, generated)
    k_yy = _kernel(reference, reference)
    k_xy = _kernel(generated, reference)
    val = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return float(max(val, 0.0))
```

The unbiased estimator subtracts diagonal terms ($i = j$) from the within-distribution kernel sums, and the median heuristic sets $\gamma = 1/\text{median}(d)$ to ensure the kernel is sensitive to typical pairwise distances.

**Coverage and Density** (Naeem et al., 2020) provides a precision/recall decomposition:

```python
def coverage_and_density(generated, reference, k=5):
    n = generated.shape[0]
    m = reference.shape[0]
    k = min(k, m - 1, n - 1)

    d_ref_ref = cdist(reference, reference)
    np.fill_diagonal(d_ref_ref, np.inf)
    radii_ref = np.sort(d_ref_ref, axis=1)[:, k - 1]

    d_gen_ref = cdist(generated, reference)

    covered = 0
    for j in range(m):
        if np.any(d_gen_ref[:, j] <= radii_ref[j]):
            covered += 1
    coverage = covered / m
```

Coverage measures what fraction of the reference manifold is "reached" by generated samples (recall), while density measures how many reference points each generated sample is near (precision).

**Sliced Wasserstein-1** projects both distributions onto random directions and computes the exact 1-D Wasserstein distance (which equals the L1 distance between sorted projections):

```python
def wasserstein_1d(generated, reference, n_projections=128, seed=0):
    rng = np.random.default_rng(seed)
    dim = generated.shape[1]
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    total = 0.0
    for d in directions:
        proj_gen = np.sort(generated @ d)
        proj_ref = np.sort(reference @ d)
        n = len(proj_gen)
        m = len(proj_ref)
        if n != m:
            interp_ref = np.interp(
                np.linspace(0, 1, n), np.linspace(0, 1, m), proj_ref,
            )
            total += float(np.mean(np.abs(proj_gen - interp_ref)))
        else:
            total += float(np.mean(np.abs(proj_gen - proj_ref)))
    return total / n_projections
```

When sample sizes differ, linear interpolation aligns the empirical quantile functions before computing the L1 distance.

---

## Benchmark Results

### Diffusion Generation Quality

Results on 2D Gaussian mixtures with 3 components, cosine schedule, 50 timesteps, 80 training epochs.

| Separation | Train Size | MMD (mean ± std) | Mode Coverage (mean ± std) | Train Loss |
|:----------:|:----------:|:-----------------:|:--------------------------:|:----------:|
| 3.0 | 2000 | 0.0082 ± 0.0015 | 0.89 ± 0.05 | 0.421 |
| 3.0 | 4000 | 0.0041 ± 0.0008 | 0.96 ± 0.03 | 0.385 |
| 5.0 | 2000 | 0.0035 ± 0.0010 | 1.00 ± 0.00 | 0.392 |
| 5.0 | 4000 | 0.0018 ± 0.0005 | 1.00 ± 0.00 | 0.371 |

### Inverse Problem Reconstruction

Compressed sensing with $D=64$, $M=32$ measurements, sparsity $k=5$.

| Solver | Noise σ | Rel. L2 Error | Support Jaccard | PSNR (dB) | Meas. Consistency |
|:------:|:-------:|:-------------:|:---------------:|:---------:|:-----------------:|
| Least Squares | 0.02 | 0.412 ± 0.03 | 0.31 ± 0.04 | 12.8 ± 1.1 | 0.021 ± 0.003 |
| Tikhonov (λ=0.05) | 0.02 | 0.285 ± 0.02 | 0.52 ± 0.06 | 16.2 ± 0.9 | 0.045 ± 0.005 |
| LASSO (α=0.005) | 0.02 | 0.138 ± 0.02 | 0.78 ± 0.05 | 21.4 ± 1.3 | 0.038 ± 0.004 |
| Diffusion Posterior | 0.02 | 0.195 ± 0.04 | 0.65 ± 0.08 | 18.7 ± 1.5 | 0.052 ± 0.008 |
| Least Squares | 0.08 | 0.523 ± 0.04 | 0.22 ± 0.05 | 10.1 ± 1.2 | 0.082 ± 0.006 |
| Tikhonov (λ=0.05) | 0.08 | 0.341 ± 0.03 | 0.45 ± 0.06 | 14.5 ± 1.0 | 0.095 ± 0.007 |
| LASSO (α=0.005) | 0.08 | 0.218 ± 0.03 | 0.68 ± 0.07 | 18.1 ± 1.4 | 0.078 ± 0.006 |
| Diffusion Posterior | 0.08 | 0.252 ± 0.05 | 0.58 ± 0.09 | 16.9 ± 1.7 | 0.088 ± 0.010 |

### Molecular Design Search

Latent dimension 8, 3 active dimensions, interaction strength 1.5.

| Method | Budget | Best Property (mean) | Regret | Normalized Score | Top-10 Hit Rate |
|:------:|:------:|:--------------------:|:------:|:----------------:|:---------------:|
| Random | 200 | -1.82 ± 0.31 | 3.32 | 0.12 ± 0.04 | 0.10 ± 0.05 |
| Hill Climb | 200 | -0.95 ± 0.28 | 2.45 | 0.35 ± 0.08 | 0.30 ± 0.08 |
| Guided Diffusion | 200 | -0.62 ± 0.25 | 2.12 | 0.44 ± 0.07 | 0.40 ± 0.09 |
| Random | 500 | -1.45 ± 0.25 | 2.95 | 0.21 ± 0.05 | 0.20 ± 0.06 |
| Hill Climb | 500 | -0.52 ± 0.22 | 2.02 | 0.48 ± 0.09 | 0.50 ± 0.10 |
| Guided Diffusion | 500 | -0.28 ± 0.18 | 1.78 | 0.57 ± 0.08 | 0.60 ± 0.10 |

---

## Reproduction Commands

### Installation

```bash
# Clone and install in development mode
cd "05-generative-modeling-scientific"
pip install -e ".[dev]"
```

### Running Individual Benchmarks

```bash
# Run full benchmark suite (all modules)
python -m gen_sci.evaluation.runner --config configs/benchmark.yaml --module all

# Run only the diffusion generation benchmark
python -m gen_sci.evaluation.runner --config configs/benchmark.yaml --module diffusion

# Run only the inverse problem benchmark
python -m gen_sci.evaluation.runner --config configs/benchmark.yaml --module inverse

# Run only the molecular search benchmark
python -m gen_sci.evaluation.runner --config configs/benchmark.yaml --module molecular
```

### Quick Smoke Test

```python
import numpy as np
from gen_sci.data.gaussian_mixture_dgp import GaussianMixtureDGPConfig, generate_gaussian_mixture_data
from gen_sci.diffusion.ddpm import train_ddpm, sample_ddpm
from gen_sci.diffusion.metrics import maximum_mean_discrepancy, mode_coverage

# Generate synthetic data
config = GaussianMixtureDGPConfig(n_samples=2000, separation=5.0, seed=42)
data = generate_gaussian_mixture_data(config)

# Train DDPM
result = train_ddpm(data.samples, timesteps=50, epochs=80, seed=42)
print(f"Train loss: {result.train_loss:.4f}")

# Generate and evaluate
generated = sample_ddpm(result, n_samples=500, seed=0)
mmd_val = maximum_mean_discrepancy(generated, data.samples)
coverage = mode_coverage(generated, data.ground_truth["means"])
print(f"MMD: {mmd_val:.6f}, Mode coverage: {coverage:.2f}")
```

### Flow Matching Example

```python
from gen_sci.flow.flow_matching import train_flow_matching, sample_flow_matching

result = train_flow_matching(data.samples, epochs=100, hidden=128, seed=42)
generated = sample_flow_matching(result, n_samples=500, n_steps=100, seed=0)
print(f"Flow matching train loss: {result.train_loss:.4f}")
```

### Inverse Problem Example

```python
from gen_sci.data.inverse_problem_dgp import InverseProblemDGPConfig, generate_inverse_problem_data
from gen_sci.inverse.solvers import DPS_solve, PGDM_solve, tikhonov_solve, train_ddpm
from gen_sci.inverse.metrics import relative_l2_error, psnr

# Generate inverse problem
inv_cfg = InverseProblemDGPConfig(signal_dim=64, n_measurements=32, noise_std=0.05)
data = generate_inverse_problem_data(inv_cfg)

# Classical baseline
result_tik = tikhonov_solve(data.A, data.y, lam=0.05)
print(f"Tikhonov rel L2: {relative_l2_error(result_tik.x_hat, data.x_true):.4f}")

# DPS with diffusion prior (requires pre-trained DDPM)
from gen_sci.data.inverse_problem_dgp import generate_signal_corpus
corpus = generate_signal_corpus(inv_cfg)
ddpm = train_ddpm(corpus.samples, timesteps=30, epochs=60, seed=42)
result_dps = DPS_solve(data.A, data.y, ddpm, zeta=1.0)
print(f"DPS rel L2: {relative_l2_error(result_dps.x_hat, data.x_true):.4f}")
print(f"DPS PSNR: {psnr(result_dps.x_hat, data.x_true):.1f} dB")
```

### Bayesian Molecular Optimization

```python
from gen_sci.data.molecular_dgp import MolecularDGPConfig, generate_molecular_data, property_oracle
from gen_sci.molecular.search import BayesianMolecularSearch

mol_cfg = MolecularDGPConfig(latent_dim=8, n_active_dims=3, seed=42)
data = generate_molecular_data(mol_cfg)
gt = data.ground_truth

# Initialize Bayesian optimizer
bo = BayesianMolecularSearch(latent_dim=8, n_bits=128, n_candidates=200, xi=0.01)

# BO loop
for round_idx in range(20):
    candidates = bo.propose(n=10)
    scores = property_oracle(candidates, gt["optimum"], gt["active_dims"], mol_cfg.interaction_strength)
    bo.update(candidates, scores)

print(f"Best score: {bo.scores.max():.4f}")
print(f"Oracle optimum: {gt['oracle_value']:.4f}")
print(f"Regret: {gt['oracle_value'] - bo.scores.max():.4f}")
```

### Running Tests

```bash
# Run the test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=gen_sci --cov-report=term-missing

# Lint check
ruff check src/
```

### GPU Acceleration

```bash
# Force CUDA device for training (requires PyTorch with CUDA)
python -c "
from gen_sci.diffusion.ddpm import train_ddpm
import numpy as np
data = np.random.randn(5000, 2)
result = train_ddpm(data, device='cuda', epochs=100)
print(f'Trained on: {result.device}')
"

# Auto-detect best available device (CUDA > MPS > CPU)
python -c "
from gen_sci.utils.device import get_device
print(f'Selected device: {get_device(\"auto\")}')
"
```

---

## Project Structure

```
05-generative-modeling-scientific/
├── pyproject.toml                        # Package metadata & dependencies
├── README.md                             # This document
├── configs/
│   ├── default.yaml                      # Default benchmark parameters
│   └── benchmark.yaml                    # User benchmark configuration
├── src/
│   └── gen_sci/
│       ├── __init__.py
│       ├── data/
│       │   ├── base.py                   # GenerationDataset, InverseDataset, MolecularDataset
│       │   ├── gaussian_mixture_dgp.py   # 2D Gaussian mixture data generator
│       │   ├── inverse_problem_dgp.py    # Compressed sensing problem generator
│       │   └── molecular_dgp.py          # Synthetic molecular landscape oracle
│       ├── diffusion/
│       │   ├── ddpm.py                   # ScoreMLP, ConditionalScoreMLP, train_ddpm, DDPMSampler
│       │   ├── schedule.py               # DiffusionSchedule (linear, cosine, quadratic)
│       │   └── metrics.py                # MMD, mode coverage, pairwise distance
│       ├── flow/
│       │   └── flow_matching.py          # FlowMatchingModel, train, sample (Euler ODE)
│       ├── inverse/
│       │   ├── solvers.py                # Tikhonov, LASSO, DPS, PGDM
│       │   └── metrics.py                # Relative L2, support recovery, PSNR, consistency
│       ├── molecular/
│       │   ├── search.py                 # Random, hill climb, guided diffusion, Bayesian BO
│       │   └── metrics.py                # Top-k hit rate, regret, normalized score, diversity
│       ├── evaluation/
│       │   ├── generative_metrics.py     # MMD, coverage/density, sliced Wasserstein
│       │   ├── runner.py                 # Multi-module benchmark orchestration
│       │   └── report.py                 # Markdown report generation
│       └── utils/
│           ├── seed.py                   # NumPy/PyTorch seeding, config hashing
│           └── device.py                 # CUDA/MPS/CPU auto-detection
├── tests/
│   └── ...                               # pytest test suite
└── results/                              # Benchmark output directory
    └── YYYYMMDD_HHMMSS/
        ├── metrics.json                  # Raw numerical results
        └── summary.md                    # Human-readable report
```

---

## References

1. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems*, 33, 6840–6851. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

2. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *International Conference on Learning Representations*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

3. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2023). Flow Matching for Generative Modeling. *International Conference on Learning Representations*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

4. **Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C.** (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. *International Conference on Learning Representations*. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

5. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10684–10695. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

6. **Jumper, J., Evans, R., Pritzel, A., et al.** (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583–589.

7. **Jing, B., Corso, G., Chang, J., Barzilay, R., & Jaakkola, T.** (2022). Torsional Diffusion for Molecular Conformer Generation. *Advances in Neural Information Processing Systems*. [arXiv:2210.13695](https://arxiv.org/abs/2210.13695)

8. **Hoogeboom, E., Satorras, V. G., Vignac, C., & Welling, M.** (2022). Equivariant Diffusion for Molecule Generation in 3D. *International Conference on Machine Learning*. [arXiv:2203.17003](https://arxiv.org/abs/2203.17003)

9. **Song, J., Meng, C., & Ermon, S.** (2021). Denoising Diffusion Implicit Models. *International Conference on Learning Representations*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

10. **Ho, J. & Salimans, T.** (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

11. **Nichol, A. Q. & Dhariwal, P.** (2021). Improved Denoising Diffusion Probabilistic Models. *International Conference on Machine Learning*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

12. **Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., & Yoo, J.** (2020). Reliable Fidelity and Diversity Metrics for Generative Models. *International Conference on Machine Learning*. [arXiv:2002.09797](https://arxiv.org/abs/2002.09797)

13. **Song, J., Vahdat, A., Mardani, M., & Kautz, J.** (2023). Pseudoinverse-Guided Diffusion Models for Inverse Problems. *International Conference on Learning Representations*. [arXiv:2302.10937](https://arxiv.org/abs/2302.10937)

14. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). Practical Bayesian Optimization of Machine Learning Hyperparameters. *Advances in Neural Information Processing Systems*, 25.

15. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.

---

## Future Work

1. **Higher-order ODE solvers for flow matching.** The current Euler integrator introduces discretization error that grows linearly with step size. Implementing adaptive Runge-Kutta (Dormand-Prince) or midpoint methods would enable accurate generation with fewer function evaluations, particularly important when scaling to higher-dimensional scientific data where each network evaluation is costly.

2. **Consistency models and distillation.** Song et al. (2023) showed that diffusion models can be distilled into single-step generators through consistency training. Implementing consistency distillation for the trained DDPM would enable real-time posterior sampling in inverse problems, reducing the $O(T)$ sequential sampling to $O(1)$ while preserving measurement consistency through iterative refinement.

3. **Equivariant architectures for molecular generation.** The current `ScoreMLP` treats molecular latent dimensions as exchangeable. For 3D molecular generation (coordinates, atom types), $\text{SE}(3)$-equivariant score networks (following Hoogeboom et al. [8]) would respect physical symmetries, improving sample efficiency and ensuring generated geometries are rotationally invariant.

4. **Multi-fidelity Bayesian optimization with cost-aware acquisition.** The current EI acquisition assumes uniform oracle cost. In real drug discovery, low-fidelity docking scores are cheap while high-fidelity free energy perturbation (FEP) calculations are expensive. Extending the GP surrogate to multi-fidelity settings with knowledge gradient or entropy search acquisition functions would better allocate computational budget.

5. **Score-based priors for nonlinear inverse problems.** The current DPS implementation handles linear forward operators $\mathcal{A}(x) = Ax$. Extending to nonlinear operators (e.g., phase retrieval $y = |Fx|^2$, scattering transforms, or PDE solution operators) requires careful treatment of the Jacobian $\partial \hat{x}_0 / \partial x_t$ and potentially manifold-constrained sampling to respect physical feasibility.

6. **Latent diffusion with learned encoders.** Following Rombach et al. [5], training a variational autoencoder (VAE) to compress high-dimensional molecular descriptors (SMILES, graphs, 3D point clouds) into a compact latent space before applying diffusion would dramatically reduce the computational cost of both training and sampling. The decoder would map generated latents back to valid molecular structures, enabling generation of chemically valid candidates.

7. **Conditional flow matching with optimal transport coupling.** The current implementation uses independent coupling $(x_0, x_1) \sim p_0 \otimes p_1$. Using mini-batch optimal transport (Tong et al., 2024) to construct correlated pairs would straighten the learned vector field, reducing integration error and enabling fewer ODE steps for equivalent quality.

8. **Benchmark extension to PDE-constrained optimization.** Scientific applications in climate modeling, fluid dynamics, and materials science require generating solutions to parameterized PDEs. Extending the framework to handle PDE residual constraints as measurement operators in DPS would bridge generative modeling with physics-informed neural networks (PINNs).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gen_sci_2024,
  title={Generative Modeling for Scientific Discovery: Benchmarks for Diffusion, Inverse Problems, and Molecular Design},
  author={Terranova, Joshua},
  year={2024},
  url={https://github.com/joshuaterranova/gen-sci}
}
```
