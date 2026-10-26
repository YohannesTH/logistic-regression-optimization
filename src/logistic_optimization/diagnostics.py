"""Diagnostics: Gradient & Hessian verification, condition numbers, classification metrics.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from .objective import objective, gradient, hessian

EPS = 1e-12


def check_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Verify analytical gradient against centered finite differences.

    Returns
    -------
    rel_error : float
        Relative L2 error between analytical and numerical gradients.
    g_analytic : np.ndarray
        Analytical gradient vector.
    g_num : np.ndarray
        Finite-difference gradient vector.
    """
    g_analytic = gradient(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
    g_num = np.zeros_like(beta)

    for j in range(len(beta)):
        e_j = np.zeros_like(beta)
        e_j[j] = eps
        f_plus = objective(X, y, beta + e_j, reg=reg, penalize_intercept=penalize_intercept)
        f_minus = objective(X, y, beta - e_j, reg=reg, penalize_intercept=penalize_intercept)
        g_num[j] = (f_plus - f_minus) / (2.0 * eps)

    diff_norm = float(np.linalg.norm(g_analytic - g_num))
    scale = float(max(np.linalg.norm(g_analytic), np.linalg.norm(g_num), 1e-8))
    rel_error = diff_norm / scale
    return rel_error, g_analytic, g_num


def check_hessian(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float = 0.0,
    penalize_intercept: bool = False,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Verify analytical Hessian against finite differences of the analytical gradient.

    Returns
    -------
    rel_error : float
        Relative Frobenius norm error between analytical and numerical Hessians.
    H_analytic : np.ndarray
        Analytical Hessian matrix.
    H_num : np.ndarray
        Finite-difference Hessian matrix.
    """
    H_analytic = hessian(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
    p = len(beta)
    H_num = np.zeros((p, p), dtype=float)

    for j in range(p):
        e_j = np.zeros_like(beta)
        e_j[j] = eps
        g_plus = gradient(X, y, beta + e_j, reg=reg, penalize_intercept=penalize_intercept)
        g_minus = gradient(X, y, beta - e_j, reg=reg, penalize_intercept=penalize_intercept)
        H_num[:, j] = (g_plus - g_minus) / (2.0 * eps)

    # Symmetrize numerical Hessian
    H_num = 0.5 * (H_num + H_num.T)

    diff_norm = float(np.linalg.norm(H_analytic - H_num, ord="fro"))
    scale = float(max(np.linalg.norm(H_analytic, ord="fro"), np.linalg.norm(H_num, ord="fro"), 1e-8))
    rel_error = diff_norm / scale
    return rel_error, H_analytic, H_num


def compute_condition_number(A: np.ndarray) -> float:
    """Compute 2-norm condition number kappa(A) = sigma_max / sigma_min."""
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float("inf")


def evaluate_classification(y_true: np.ndarray, probas: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard classification diagnostics on binary predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probas = np.clip(np.asarray(probas, dtype=float), EPS, 1.0 - EPS)
    preds = (probas >= threshold).astype(int)

    # Handle single-class edge cases gracefully in roc_auc
    try:
        roc_auc = float(roc_auc_score(y_true, probas))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": roc_auc,
        "log_loss": float(log_loss(y_true, probas)),
    }
