"""Synthetic Data Generation for Controlled Numerical Experiments.

Enables precise control over:
- Condition number kappa(X)
- Pairwise feature correlation rho
- Multi-scale feature variances
- Linear separability (to test complete separation and finite MLE existence)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from .objective import sigmoid


def generate_ill_conditioned_data(
    n_samples: int = 500,
    n_features: int = 10,
    condition_number: float = 100.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic logistic regression dataset with controlled condition number kappa(X).

    Parameters
    ----------
    n_samples : int, default 500
    n_features : int, default 10
    condition_number : float, default 100.0
        Desired 2-norm condition number of the feature matrix X.
    random_state : int, default 42

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    true_beta : np.ndarray, shape (n_features,)
    """
    rng = np.random.default_rng(random_state)
    # Generate random orthogonal matrices via SVD
    A = rng.standard_normal((n_samples, n_features))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)

    # Logarithmically spaced singular values from 1.0 down to 1.0 / condition_number
    s = np.geomspace(1.0, 1.0 / condition_number, num=n_features)
    X = U @ np.diag(s) @ Vt

    # Standardize variance of X
    X = X * np.sqrt(n_samples)

    true_beta = rng.standard_normal(n_features)
    probs = sigmoid(X @ true_beta)
    y = (rng.uniform(size=n_samples) < probs).astype(float)
    return X, y, true_beta


def generate_correlated_data(
    n_samples: int = 500,
    n_features: int = 10,
    rho: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate features with controlled pairwise correlation rho.

    Covariance matrix Sigma has Sigma_ii = 1 and Sigma_ij = rho (equicorrelation).
    """
    rng = np.random.default_rng(random_state)
    Sigma = np.full((n_features, n_features), rho)
    np.fill_diagonal(Sigma, 1.0)

    # Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n_samples, n_features))
    X = Z @ L.T

    true_beta = rng.standard_normal(n_features)
    probs = sigmoid(X @ true_beta)
    y = (rng.uniform(size=n_samples) < probs).astype(float)
    return X, y, true_beta


def generate_multiscale_data(
    n_samples: int = 500,
    n_features: int = 10,
    scale_range: Tuple[float, float] = (1e-2, 1e4),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate features where each column has variance spanning several orders of magnitude."""
    rng = np.random.default_rng(random_state)
    X_std = rng.standard_normal((n_samples, n_features))
    scales = np.geomspace(scale_range[0], scale_range[1], num=n_features)
    X = X_std * scales

    true_beta = rng.standard_normal(n_features) / scales
    probs = sigmoid(X @ true_beta)
    y = (rng.uniform(size=n_samples) < probs).astype(float)
    return X, y, true_beta


def generate_separable_data(
    n_samples: int = 300,
    n_features: int = 5,
    margin: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate strictly linearly separable data with a guaranteed geometric margin.

    Used to demonstrate failure of unregularized MLE (divergence of ||beta|| -> inf).
    """
    rng = np.random.default_rng(random_state)
    separating_w = rng.standard_normal(n_features)
    separating_w /= np.linalg.norm(separating_w)

    X_list = []
    y_list = []
    while len(X_list) < n_samples:
        pt = rng.standard_normal(n_features)
        dist = pt @ separating_w
        if dist > margin:
            X_list.append(pt)
            y_list.append(1.0)
        elif dist < -margin:
            X_list.append(pt)
            y_list.append(0.0)

    return np.array(X_list), np.array(y_list), separating_w
