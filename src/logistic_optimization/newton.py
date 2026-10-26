"""Newton's Method and Damped Newton's Method (with Armijo Backtracking).

Solves the second-order Newton system:
    nabla^2 J(beta) * s = nabla J(beta)
and updates:
    beta_{k+1} = beta_k - alpha * s
where alpha = 1 for pure Newton, or chosen via Armijo line search for damped Newton.
Tracks the Newton decrement: lambda(beta) = sqrt(grad^T * s).
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Any
import numpy as np

from .objective import objective, gradient, hessian
from .line_search import armijo_backtracking


def newton_method(
    X: np.ndarray,
    y: np.ndarray,
    beta0: np.ndarray | None = None,
    damping: bool = True,
    max_iter: int = 100,
    tol: float = 1e-7,
    decrement_tol: float = 1e-12,
    reg: float = 0.0,
    penalize_intercept: bool = False,
    c1: float = 1e-4,
    rho: float = 0.5,
    min_step: float = 1e-14,
) -> Tuple[np.ndarray, Dict[str, Any], List[float]]:
    """Fit logistic regression via Newton's method.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Binary response vector.
    beta0 : np.ndarray, optional
        Initial parameter vector. Defaults to zeros(p).
    damping : bool, default True
        If True, use Armijo backtracking line search (damped Newton).
        If False, use standard unit step size alpha=1 (pure Newton).
    max_iter : int, default 100
        Maximum Newton iterations.
    tol : float, default 1e-7
        Stopping tolerance on ||grad J(beta)||_inf.
    decrement_tol : float, default 1e-12
        Stopping tolerance on 0.5 * lambda(beta)^2.
    reg : float, default 0.0
        L2 regularization parameter lambda.
    penalize_intercept : bool, default False
        Whether to penalize the intercept.
    c1 : float, default 1e-4
        Armijo sufficient decrease parameter.
    rho : float, default 0.5
        Backtracking contraction factor.
    min_step : float, default 1e-14
        Minimum allowable step size.

    Returns
    -------
    beta : np.ndarray
        Final parameter vector.
    info : dict
        Optimization diagnostics: iterations, objective, gradient_inf_norm,
        converged, seconds, n_fev, n_gev, n_hev, condition_numbers,
        newton_decrements, step_sizes.
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
    condition_numbers = []
    newton_decrements = []

    n_fev = 1
    n_gev = 0
    n_hev = 0

    start_time = time.perf_counter()
    converged = False

    for it in range(max_iter):
        g = gradient(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
        n_gev += 1
        g_inf = float(np.linalg.norm(g, ord=np.inf))
        grad_norms.append(g_inf)

        if g_inf < tol:
            converged = True
            break

        H = hessian(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
        n_hev += 1

        # Track condition number of the Hessian
        try:
            cond_H = float(np.linalg.cond(H))
        except Exception:
            cond_H = float("inf")
        condition_numbers.append(cond_H)

        # Solve linear system H * s = g
        try:
            s = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback for ill-conditioned or singular Hessian
            s = np.linalg.lstsq(H, g, rcond=None)[0]

        # Newton decrement squared: lambda^2 = g^T * s
        lambda_sq = float(g @ s)
        newton_decrements.append(0.5 * max(0.0, lambda_sq))

        # Check affine-invariant stopping condition: 0.5 * lambda^2 < decrement_tol
        if lambda_sq >= 0 and 0.5 * lambda_sq < decrement_tol:
            converged = True
            break

        # Check descent direction: d = -s
        direction = -s
        directional_derivative = -lambda_sq

        if directional_derivative >= 0 or not np.isfinite(lambda_sq):
            # If not a descent direction (e.g. non-positive Hessian near boundary),
            # fall back to steepest descent direction
            direction = -g
            directional_derivative = -float(g @ g)

        if damping:
            alpha, f_next, evals, success = armijo_backtracking(
                fn=loss_fn,
                beta=beta,
                direction=direction,
                grad=g,
                f0=f_current,
                alpha_init=1.0,
                c1=c1,
                rho=rho,
                min_step=min_step,
            )
            n_fev += evals
            if not success or not np.isfinite(f_next):
                break
            step_size = alpha
        else:
            step_size = 1.0
            candidate_beta = beta + step_size * direction
            f_next = loss_fn(candidate_beta)
            n_fev += 1
            if not np.isfinite(f_next):
                # Pure Newton diverged
                break

        beta = beta + step_size * direction
        f_current = f_next
        history.append(f_current)
        step_sizes.append(step_size)

    elapsed = time.perf_counter() - start_time

    final_g = gradient(X, y, beta, reg=reg, penalize_intercept=penalize_intercept)
    final_g_inf = float(np.linalg.norm(final_g, ord=np.inf))
    final_g_2 = float(np.linalg.norm(final_g, ord=2))

    method_name = "Newton (Armijo backtracking)" if damping else "Pure Newton"
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
        "n_hev": n_hev,
        "condition_numbers": condition_numbers,
        "newton_decrements": newton_decrements,
        "step_sizes": step_sizes,
        "grad_norms": grad_norms,
    }
    return beta, info, history
