"""Unit tests for analytical Hessian via finite differences and positive semidefiniteness."""

import numpy as np
import pytest
from logistic_optimization.objective import hessian
from logistic_optimization.diagnostics import check_hessian


@pytest.mark.parametrize("p", [2, 5, 8])
@pytest.mark.parametrize("reg", [0.0, 1e-4, 1.0])
@pytest.mark.parametrize("penalize_intercept", [False, True])
def test_hessian_finite_differences(p, reg, penalize_intercept):
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, p))
    X[:, 0] = 1.0
    y = rng.integers(0, 2, size=n).astype(float)
    beta = rng.standard_normal(p) * 0.5

    rel_error, H_ana, H_num = check_hessian(
        X, y, beta, reg=reg, penalize_intercept=penalize_intercept, eps=1e-6
    )
    assert rel_error < 1e-5, f"Hessian check failed with rel_error {rel_error:e}"


def test_hessian_positive_semidefinite():
    """Verify that the Hessian is positive semidefinite everywhere."""
    rng = np.random.default_rng(999)
    n, p = 100, 6
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, size=n).astype(float)

    for _ in range(5):
        beta = rng.standard_normal(p) * 2.0
        H = hessian(X, y, beta, reg=0.0)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-12), f"Negative eigenvalue found: {np.min(eigenvalues)}"
