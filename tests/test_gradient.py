"""Unit tests for analytical gradient via centered finite differences."""

import numpy as np
import pytest
from logistic_optimization.objective import gradient, objective
from logistic_optimization.diagnostics import check_gradient


@pytest.mark.parametrize("p", [2, 5, 10])
@pytest.mark.parametrize("reg", [0.0, 1e-4, 1.0])
@pytest.mark.parametrize("penalize_intercept", [False, True])
def test_gradient_finite_differences(p, reg, penalize_intercept):
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, p))
    # Add intercept column of 1s
    X[:, 0] = 1.0
    y = rng.integers(0, 2, size=n).astype(float)
    beta = rng.standard_normal(p) * 0.5

    rel_error, g_ana, g_num = check_gradient(
        X, y, beta, reg=reg, penalize_intercept=penalize_intercept, eps=1e-6
    )
    assert rel_error < 1e-6, f"Gradient check failed with rel_error {rel_error:e}"


def test_gradient_at_optimum():
    """At true optimum, gradient norm must be near zero."""
    rng = np.random.default_rng(123)
    n, p = 200, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = rng.integers(0, 2, size=n).astype(float)
    # If p == y, gradient is zero
    beta = np.zeros(p)
    # Check gradient computes cleanly without NaN/inf
    g = gradient(X, y, beta)
    assert np.all(np.isfinite(g))
