"""Mathematical foundations: Likelihood, Gradient, Hessian, and Lipschitz constant.

Supports binary cross-entropy loss with optional L2 regularization.
Numerical operations use stable logaddexp and two-branch sigmoid formulations.
"""

from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Numerically stable element-wise sigmoid function.
    
    sigma(z) = 1 / (1 + exp(-z)) for z >= 0
             = exp(z) / (1 + exp(z)) for z < 0
    Prevents overflow for large negative and positive logits.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _get_penalty_matrix(dim: int, reg: float, penalize_intercept: bool) -> np.ndarray:
    """Constructs diagonal regularization matrix R."""
    if reg <= 0.0:
        return np.zeros((dim, dim))
    mask = np.ones(dim, dtype=float)
    if not penalize_intercept:
        mask[0] = 0.0
    return reg * np.diag(mask)


def objective(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
) -> float:
    """Compute mean negative log-likelihood (binary cross-entropy) + 0.5 * reg * beta^T R beta.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix (may include leading column of 1s for intercept).
    y : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    beta : np.ndarray, shape (p,)
        Coefficient vector.
    reg : float, default 0.0
        L2 regularization strength (lambda).
    penalize_intercept : bool, default False
        Whether to penalize beta[0].
        
    Returns
    -------
    float : Objective function value J(beta).
    """
    z = X @ beta
    # Stable negative log-likelihood: log(1 + exp(z)) - y * z
    nll = np.mean(np.logaddexp(0.0, z) - y * z)
    if reg > 0.0:
        mask = np.ones_like(beta)
        if not penalize_intercept:
            mask[0] = 0.0
        penalty = 0.5 * reg * np.sum((mask * beta) ** 2)
        return float(nll + penalty)
    return float(nll)


def gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Compute analytical gradient of the regularized objective.
    
    grad J(beta) = (1/n) * X^T (p - y) + reg * R * beta
    where p = sigmoid(X @ beta).
    """
    n = len(y)
    p = sigmoid(X @ beta)
    g = (X.T @ (p - y)) / n
    if reg > 0.0:
        mask = np.ones_like(beta)
        if not penalize_intercept:
            mask[0] = 0.0
        g = g + reg * (mask * beta)
    return g


def hessian(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Compute analytical Hessian matrix of the regularized objective.
    
    H = (1/n) * X^T W X + reg * R
    where W = diag(p_i * (1 - p_i)).
    """
    n = len(y)
    p = sigmoid(X @ beta)
    w = p * (1.0 - p)
    # Vectorized: (X.T * w) @ X is mathematically equivalent to X^T W X
    H = (X.T * w) @ X / n
    if reg > 0.0:
        mask = np.ones_like(beta)
        if not penalize_intercept:
            mask[0] = 0.0
        H = H + reg * np.diag(mask)
    return H


def lipschitz_constant(
    X: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
) -> float:
    """Compute upper bound on the Lipschitz constant L of the gradient.
    
    Since p_i * (1 - p_i) <= 1/4 for all logits,
    ||grad^2 J(beta)||_2 <= (1 / (4n)) * lambda_max(X^T X) + reg = L.
    
    Guarantees gradient descent convergence for step sizes eta < 2 / L.
    """
    n = X.shape[0]
    # Compute largest singular value of X / sqrt(n)
    s_max = np.linalg.norm(X, ord=2)
    L_loss = (s_max ** 2) / (4.0 * n)
    return float(L_loss + reg)
