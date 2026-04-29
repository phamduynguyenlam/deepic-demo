# moead_ego_baseline.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats import qmc


# -------------------------
# Utilities
# -------------------------

def lhs_sample(lower, upper, n_samples, seed=0):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    sampler = qmc.LatinHypercube(d=len(lower), seed=seed)
    u = sampler.random(n_samples)
    return qmc.scale(u, lower, upper).astype(np.float64)


def normalize_y(Y, eps=1e-12):
    y_min = Y.min(axis=0)
    y_rng = np.maximum(Y.max(axis=0) - y_min, eps)
    return (Y - y_min) / y_rng, y_min, y_rng


def weighted_tchebycheff(Y_norm, weights, ideal=None, eps=1e-6):
    """
    Scalarization for minimization.
    Y_norm: (N, m)
    weights: (m,)
    """
    if ideal is None:
        ideal = Y_norm.min(axis=0)
    w = np.maximum(weights, eps)
    return np.max(w * np.abs(Y_norm - ideal), axis=1)


def expected_improvement_min(mu, sigma, best, xi=0.0):
    """
    EI for minimization.
    improvement = best - f(x)
    """
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def make_weights_2obj(n_weights=40):
    """
    Weight vectors for 2-objective MOEA/D.
    """
    a = np.linspace(0.0, 1.0, n_weights)
    W = np.stack([a, 1.0 - a], axis=1)
    W = np.clip(W, 1e-6, 1.0)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def fit_gp(X, y, seed=0):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(X.shape[1]), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )
    gp.fit(X, y)
    return gp


# -------------------------
# MOEA/D-EGO main
# -------------------------

def run_moead_ego(
    problem,
    dim=30,
    init_size=80,
    max_fe=120,
    n_weights=40,
    de_maxiter=30,
    seed=0,
    verbose=True,
):
    """
    problem must expose:
        problem.lower: shape (dim,)
        problem.upper: shape (dim,)
        problem.evaluate(X): returns shape (N, m)

    Assumes minimization.
    """

    rng = np.random.default_rng(seed)

    lower = np.asarray(problem.lower, dtype=np.float64)
    upper = np.asarray(problem.upper, dtype=np.float64)

    # 1. Initial expensive evaluations
    X = lhs_sample(lower, upper, init_size, seed=seed)
    Y = problem.evaluate(X).astype(np.float64)

    fe = init_size
    hv_history = []

    # Only written for 2 objectives here
    W = make_weights_2obj(n_weights=n_weights)

    while fe < max_fe:
        # 2. Normalize objectives for scalarization
        Y_norm, y_min, y_rng = normalize_y(Y)
        ideal = Y_norm.min(axis=0)

        # 3. Randomly choose one MOEA/D subproblem / weight
        w = W[rng.integers(0, len(W))]

        # 4. Build scalar target g(x | w)
        g_train = weighted_tchebycheff(Y_norm, w, ideal=ideal)

        # 5. Fit GP on scalarized objective
        gp = fit_gp(X, g_train, seed=seed + fe)

        best_g = float(np.min(g_train))

        # 6. Optimize EI using differential evolution
        def neg_ei(x):
            x = np.asarray(x, dtype=np.float64).reshape(1, -1)
            mu, std = gp.predict(x, return_std=True)
            ei = expected_improvement_min(mu[0], std[0], best_g)
            return -float(ei)

        bounds = list(zip(lower, upper))

        result = differential_evolution(
            neg_ei,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=8,
            polish=False,
            seed=seed + fe,
            updating="immediate",
            workers=1,
        )

        x_new = np.asarray(result.x, dtype=np.float64).reshape(1, -1)

        # 7. Avoid exact duplicate
        if np.min(np.linalg.norm(X - x_new, axis=1)) < 1e-8:
            x_new = lhs_sample(lower, upper, 1, seed=seed + 100000 + fe)

        y_new = problem.evaluate(x_new).astype(np.float64)

        X = np.vstack([X, x_new])
        Y = np.vstack([Y, y_new])
        fe += 1

        if verbose:
            print(
                f"FE {fe:03d}/{max_fe} | "
                f"w={np.round(w, 3)} | "
                f"selected_y={np.round(y_new[0], 6)} | "
                f"best_f={np.round(Y.min(axis=0), 6)}"
            )

    return {
        "archive_x": X,
        "archive_y": Y,
        "fe": fe,
    }