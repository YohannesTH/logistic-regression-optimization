"""Armijo Backtracking Line Search.

Finds a step size alpha that satisfies the Armijo (sufficient decrease) condition:
    f(beta + alpha * d) <= f(beta) + c1 * alpha * (grad^T d)
where d is a descent direction (grad^T d < 0).
"""

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np


def armijo_backtracking(
    fn: Callable[[np.ndarray], float],
    beta: np.ndarray,
    direction: np.ndarray,
    grad: np.ndarray,
    f0: float | None = None,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    min_step: float = 1e-14,
    max_evals: int = 50,
) -> Tuple[float, float, int, bool]:
    """Perform Armijo backtracking line search along direction.

    Parameters
    ----------
    fn : Callable[[np.ndarray], float]
        Objective function mapping parameter vector to scalar loss.
    beta : np.ndarray
        Current parameter iterate.
    direction : np.ndarray
        Search direction vector d.
    grad : np.ndarray
        Gradient vector at current iterate.
    f0 : float, optional
        Current objective value fn(beta). Computed if None.
    alpha_init : float, default 1.0
        Initial trial step size.
    c1 : float, default 1e-4
        Sufficient decrease constant in (0, 1).
    rho : float, default 0.5
        Contraction factor in (0, 1).
    min_step : float, default 1e-14
        Minimum allowable step size before termination.
    max_evals : int, default 50
        Maximum objective function evaluations.

    Returns
    -------
    alpha : float
        Selected step size.
    f_val : float
        Objective value at beta + alpha * direction.
    num_evals : int
        Number of function evaluations performed.
    success : bool
        Whether sufficient decrease was satisfied.
    """
    if f0 is None:
        f0 = fn(beta)

    directional_derivative = float(grad @ direction)
    if directional_derivative >= 0:
        # Not a descent direction!
        return 0.0, f0, 0, False

    alpha = float(alpha_init)
    evals = 0
    f_candidate = f0

    while evals < max_evals and alpha >= min_step:
        candidate_beta = beta + alpha * direction
        f_candidate = fn(candidate_beta)
        evals += 1

        if np.isfinite(f_candidate) and f_candidate <= f0 + c1 * alpha * directional_derivative:
            return alpha, float(f_candidate), evals, True

        alpha *= rho

    # If line search failed to satisfy sufficient decrease
    return alpha, float(f_candidate), evals, False
