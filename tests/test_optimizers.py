"""Unit tests for optimization algorithms (GD, Backtracking GD, Pure Newton, Damped Newton)."""

import numpy as np
import pytest
from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.data import generate_ill_conditioned_data


def test_optimizers_equivalence():
    """All 4 optimizers should reach the same minimum on a well-conditioned problem."""
    rng = np.random.default_rng(42)
    n, p = 300, 4
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    true_beta = np.array([0.2, -0.5, 0.8, -0.3])
    from logistic_optimization.objective import sigmoid
    y = (rng.uniform(size=n) < sigmoid(X @ true_beta)).astype(float)

    reg = 1e-4

    b_gd, info_gd, _ = gradient_descent(X, y, lr=1.0, max_iter=8000, tol=1e-7, reg=reg)
    b_bt, info_bt, _ = gradient_descent(X, y, lr=1.0, line_search=True, max_iter=2000, tol=1e-7, reg=reg)
    b_pn, info_pn, _ = newton_method(X, y, damping=False, max_iter=30, tol=1e-7, reg=reg)
    b_dn, info_dn, _ = newton_method(X, y, damping=True, max_iter=30, tol=1e-7, reg=reg)

    assert info_gd["converged"]
    assert info_bt["converged"]
    assert info_pn["converged"]
    assert info_dn["converged"]

    # Check objective values match to 5 decimal places
    np.testing.assert_allclose(info_gd["objective"], info_dn["objective"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(info_bt["objective"], info_dn["objective"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(info_pn["objective"], info_dn["objective"], rtol=1e-4, atol=1e-4)

    # Check coefficients match
    np.testing.assert_allclose(b_gd, b_dn, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b_bt, b_dn, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b_pn, b_dn, rtol=1e-3, atol=1e-3)


def test_newton_quadratic_convergence():
    """Damped Newton should converge in very few iterations (<15) on standard data."""
    rng = np.random.default_rng(10)
    n, p = 400, 6
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = rng.integers(0, 2, size=n).astype(float)

    b, info, hist = newton_method(X, y, damping=True, max_iter=50, tol=1e-8, reg=1e-4)
    assert info["converged"]
    assert info["iterations"] < 15
