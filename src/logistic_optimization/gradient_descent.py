"""Batch Gradient Descent with optional Armijo Backtracking Line Search.
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Any
import numpy as np

from .objective import objective, gradient
from .line_search import armijo_backtracking


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    beta0: np.ndarray | None = None,
    lr: float = 0.1,
    line_search: bool = False,
    max_iter: int = 5000,
    tol: float = 1e-7,
    reg: float = 0.0,
    penalize_intercept: bool = False,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any], List[float]]:
    """Fit logistic regression via Gradient Descent.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Binary response vector.
    beta0 : np.ndarray, optional
        Initial parameter vector. Defaults to zeros(p).
    lr : float, default 0.1
        Fixed learning rate (or initial step for backtracking).
    line_search : bool, default False
        If True, use Armijo backtracking line search at each iteration.
    max_iter : int, default 5000
        Maximum number of iterations.
    tol : float, default 1e-7
        Stopping tolerance on ||grad J(beta)||_inf.
    reg : float, default 0.0
        L2 regularization parameter lambda.
    penalize_intercept : bool, default False
        Whether to penalize the intercept term.
    c1 : float, default 1e-4
        Armijo sufficient decrease parameter.
    rho : float, default 0.5
        Contraction parameter for backtracking.

    Returns
    -------
    beta : np.ndarray
        Final parameter vector.
    info : dict
        Optimization diagnostics: iterations, objective, gradient_inf_norm,
        gradient_2_norm, converged, seconds, n_fev, n_gev, step_sizes.
    history : list of float
        Objective function value history per iteration.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float) if beta0 is None else np.array(beta0, dtype=float).copy()

    def loss_fn(b: np.ndarray) -> float:
        return objective(X, y, b, reg=reg, penalize_intercept=penalize_intercept)

    f_current = loss_fn(beta)
    history = [f_current]
    grad_norms = []
    step_sizes = []
    n_fev = 1
    n_gev = 0

    start_time = time.perf_counter()
    converged = False

    for it in range(max_iter):
        g = gradient(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
        n_gev += 1
        g_inf = float(np.linalg.norm(g, ord=np.inf))
        g_2 = float(np.linalg.norm(g, ord=2))
        grad_norms.append(g_inf)

        if g_inf < tol:
            converged = True
            break

        direction = -g

        if line_search:
            alpha, f_next, evals, success = armijo_backtracking(
                fn=loss_fn,
                beta=beta,
                direction=direction,
                grad=g,
                f0=f_current,
                alpha_init=lr,
                c1=c1,
                rho=rho,
            )
            n_fev += evals
            if not success or not np.isfinite(f_next):
                # Stalled or failed line search
                break
            step_size = alpha
        else:
            step_size = lr
            beta_candidate = beta + step_size * direction
            f_next = loss_fn(beta_candidate)
            n_fev += 1
            if not np.isfinite(f_next):
                # Diverged due to excessive learning rate
                break

        beta = beta + step_size * direction
        f_current = f_next
        history.append(f_current)
        step_sizes.append(step_size)

    elapsed = time.perf_counter() - start_time

    # Final gradient evaluation for definitive report
    final_g = gradient(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
    final_g_inf = float(np.linalg.norm(final_g, ord=np.inf))
    final_g_2 = float(np.linalg.norm(final_g, ord=2))

    method_name = "Gradient descent (Armijo)" if line_search else f"Gradient descent (lr={lr})"
    info = {
        "method": method_name,
        "iterations": len(history) - 1,
        "objective": float(history[-1]),
        "gradient_inf_norm": final_g_inf,
        "gradient_2_norm": final_g_2,
        "converged": bool(converged or final_g_inf < tol),
        "seconds": elapsed,
        "n_fev": n_fev,
        "n_gev": n_gev,
        "step_sizes": step_sizes,
        "grad_norms": grad_norms,
    }
    return beta, info, history
