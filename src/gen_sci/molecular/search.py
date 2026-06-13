"""Molecular design search algorithms.

Includes random search, hill-climbing, diffusion-guided search, and
Bayesian optimisation with a Gaussian Process surrogate over extended
connectivity fingerprints (ECFP-like random projections).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from gen_sci.data.molecular_dgp import property_oracle
from gen_sci.diffusion.ddpm import sample_ddpm, train_ddpm

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a molecular design search run.

    Attributes:
        best_latent: Latent vector achieving the highest property.
        best_property: Property value at ``best_latent``.
        all_properties: All evaluated property values.
        method: Name of the search algorithm.
        n_evaluations: Total oracle calls consumed.
    """

    best_latent: np.ndarray
    best_property: float
    all_properties: np.ndarray
    method: str
    n_evaluations: int


# ---------------------------------------------------------------------------
# Baseline search methods
# ---------------------------------------------------------------------------

def random_search(
    latent_dim: int,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    seed: int = 42,
) -> SearchResult:
    """Uniform random search in latent space.

    Args:
        latent_dim: Dimensionality of the latent space.
        n_evaluations: Number of oracle queries.
        optimum: True optimum vector (for the oracle).
        active_dims: Indices of active latent dimensions.
        interaction_strength: Oracle interaction coefficient.
        seed: Random seed.

    Returns:
        ``SearchResult`` with the best random candidate.
    """
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n_evaluations, latent_dim))
    props = property_oracle(latents, optimum, active_dims, interaction_strength)
    best_idx = int(np.argmax(props))
    return SearchResult(
        best_latent=latents[best_idx],
        best_property=float(props[best_idx]),
        all_properties=props,
        method="random",
        n_evaluations=n_evaluations,
    )


def hill_climb_search(
    latent_dim: int,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    step_size: float = 0.3,
    seed: int = 42,
) -> SearchResult:
    """Greedy local search with Gaussian perturbations.

    Args:
        latent_dim: Dimensionality of the latent space.
        n_evaluations: Number of oracle queries.
        optimum: True optimum vector (for the oracle).
        active_dims: Indices of active latent dimensions.
        interaction_strength: Oracle interaction coefficient.
        step_size: Standard deviation of the perturbation noise.
        seed: Random seed.

    Returns:
        ``SearchResult`` tracking the best-seen candidate.
    """
    rng = np.random.default_rng(seed)
    current = rng.standard_normal(latent_dim)
    current_prop = float(
        property_oracle(current[None], optimum, active_dims, interaction_strength)[0]
    )
    props = [current_prop]
    latents = [current.copy()]

    for _ in range(n_evaluations - 1):
        proposal = current + step_size * rng.standard_normal(latent_dim)
        prop = float(property_oracle(proposal[None], optimum, active_dims, interaction_strength)[0])
        if prop >= current_prop:
            current, current_prop = proposal, prop
        props.append(prop)
        latents.append(current.copy())

    props_arr = np.array(props)
    best_idx = int(np.argmax(props_arr))
    return SearchResult(
        best_latent=latents[best_idx],
        best_property=float(props_arr[best_idx]),
        all_properties=props_arr,
        method="hill_climb",
        n_evaluations=n_evaluations,
    )


def guided_diffusion_search(
    train_latents: np.ndarray,
    n_evaluations: int,
    optimum: np.ndarray,
    active_dims: np.ndarray,
    interaction_strength: float,
    timesteps: int = 40,
    epochs: int = 60,
    seed: int = 42,
    device: str = "cpu",
) -> SearchResult:
    """Train latent diffusion prior, sample candidates, rank by oracle property.

    Args:
        train_latents: Training corpus for the diffusion prior ``(N, D)``.
        n_evaluations: Number of candidate samples to generate.
        optimum: True optimum vector (for the oracle).
        active_dims: Active latent dimensions.
        interaction_strength: Oracle interaction coefficient.
        timesteps: Diffusion schedule length.
        epochs: DDPM training epochs.
        seed: Random seed.
        device: Compute device.

    Returns:
        ``SearchResult`` with the best diffusion-generated candidate.
    """
    ddpm = train_ddpm(
        train_latents,
        timesteps=timesteps,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    candidates = sample_ddpm(ddpm, n_samples=n_evaluations, seed=seed + 1)
    props = property_oracle(candidates, optimum, active_dims, interaction_strength)
    best_idx = int(np.argmax(props))
    return SearchResult(
        best_latent=candidates[best_idx],
        best_property=float(props[best_idx]),
        all_properties=props,
        method="guided_diffusion",
        n_evaluations=n_evaluations,
    )


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _ecfp_features(
    latents: np.ndarray,
    n_bits: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """Compute ECFP-like random-projection fingerprints.

    Since we work in a *synthetic* latent space (no real SMILES), we
    approximate extended-connectivity fingerprints with a fixed random
    hash: each bit is the sign of a random linear projection plus a bias.

    Args:
        latents: Latent vectors of shape ``(N, D)``.
        n_bits: Length of the binary fingerprint.
        seed: Seed controlling the random projection matrix.

    Returns:
        Binary fingerprint array of shape ``(N, n_bits)`` with values
        in ``{0, 1}``.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((latents.shape[1], n_bits))
    b = rng.standard_normal(n_bits)
    return (latents @ W + b > 0).astype(np.float64)


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement (EI) acquisition function.

    .. math::
        \\text{EI}(x) = (\\mu(x) - f^* - \\xi)\\, \\Phi(Z)
                       + \\sigma(x)\\, \\phi(Z)

    where ``Z = (mu - f* - xi) / sigma``.

    Args:
        mu: GP predictive means of shape ``(N,)``.
        sigma: GP predictive standard deviations of shape ``(N,)``.
        f_best: Best observed function value so far.
        xi: Exploration–exploitation trade-off parameter.

    Returns:
        EI values of shape ``(N,)``.
    """
    improvement = mu - f_best - xi
    safe_sigma = np.maximum(sigma, 1e-12)
    Z = improvement / safe_sigma
    ei = improvement * norm.cdf(Z) + safe_sigma * norm.pdf(Z)
    ei[sigma < 1e-12] = 0.0
    return ei


def random_molecule_mutate(
    latent: np.ndarray,
    step_size: float = 0.3,
    n_children: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Generate mutant latent vectors by Gaussian perturbation.

    Args:
        latent: Parent latent vector of shape ``(D,)``.
        step_size: Standard deviation of the perturbation.
        n_children: Number of mutant offspring.
        seed: Random seed.

    Returns:
        Mutant latent vectors of shape ``(n_children, D)``.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_children, latent.shape[0]))
    return latent + step_size * noise


# ---------------------------------------------------------------------------
# Bayesian Molecular Search
# ---------------------------------------------------------------------------

@dataclass
class BayesianMolecularSearch:
    """Bayesian optimisation over molecular latent space.

    Uses a Gaussian Process surrogate fitted on ECFP-like fingerprints
    with an Expected Improvement acquisition function to propose new
    candidates at each round.

    Args:
        latent_dim: Dimensionality of the molecular latent space.
        n_bits: Fingerprint length for the GP features.
        n_candidates: Number of random candidates evaluated per
            ``propose`` call.
        xi: EI exploration parameter.
        step_size: Perturbation size for mutation-based proposals.
        seed: Master random seed.

    Attributes:
        latents: Observed latent vectors ``(K, D)``.
        scores: Corresponding property scores ``(K,)``.
        gp: Fitted ``GaussianProcessRegressor`` (or ``None``).
    """

    latent_dim: int
    n_bits: int = 128
    n_candidates: int = 200
    xi: float = 0.01
    step_size: float = 0.3
    seed: int = 0
    latents: np.ndarray = field(init=False)
    scores: np.ndarray = field(init=False)
    gp: GaussianProcessRegressor | None = field(init=False, default=None)
    _call_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.latents = np.empty((0, self.latent_dim), dtype=np.float64)
        self.scores = np.empty(0, dtype=np.float64)

    def update(self, latents: np.ndarray, scores: np.ndarray) -> None:
        """Append new observations and refit the GP surrogate.

        Args:
            latents: New latent vectors ``(K, D)``.
            scores: Corresponding oracle scores ``(K,)``.
        """
        if latents.ndim == 1:
            latents = latents[None, :]
        self.latents = np.vstack([self.latents, latents]) if self.latents.shape[0] > 0 else latents
        self.scores = np.concatenate([self.scores, np.atleast_1d(scores)])
        X = _ecfp_features(self.latents, n_bits=self.n_bits, seed=self.seed)
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
        self.gp.fit(X, self.scores)
        logger.debug(
            "GP refitted on %d observations, best=%.4f",
            len(self.scores),
            float(self.scores.max()),
        )

    def propose(self, n: int = 1) -> np.ndarray:
        """Propose the next *n* latent candidates via EI maximisation.

        If no data has been observed yet, returns random candidates.
        Otherwise generates a pool of random + mutated candidates,
        evaluates EI under the GP, and returns the top *n*.

        Args:
            n: Number of proposals to return.

        Returns:
            Candidate latent vectors of shape ``(n, D)``.
        """
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


SEARCH_METHODS = {
    "random": random_search,
    "hill_climb": hill_climb_search,
    "guided_diffusion": guided_diffusion_search,
}
