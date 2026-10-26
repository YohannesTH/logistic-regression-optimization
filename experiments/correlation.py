"""Experiment C: Feature Correlation and Hessian Conditioning.

Investigates:
1. Impact of pairwise feature correlation rho in [0.0, 0.99] on Hessian condition number.
2. Degradation of first-order gradient descent iterations vs robustness of second-order Newton.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_optimization.data import generate_correlated_data
from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.objective import hessian, lipschitz_constant
from logistic_optimization.diagnostics import compute_condition_number


def run_experiment_correlation(out_dir: Path) -> pd.DataFrame:
    rhos = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]
    n_samples = 600
    n_features = 8
    max_iter_gd = 8000
    tol = 1e-6
    reg = 1e-4

    records = []

    for rho in rhos:
        X, y, _ = generate_correlated_data(
            n_samples=n_samples, n_features=n_features, rho=rho, random_state=42
        )
        Xa = np.column_stack([np.ones(n_samples), X])
        cond_X = compute_condition_number(Xa)

        # Initial Hessian condition number at beta=0
        H0 = hessian(Xa, y, np.zeros(Xa.shape[1]), reg=reg, penalize_intercept=False)
        cond_H0 = compute_condition_number(H0)

        L = lipschitz_constant(Xa, reg=reg, penalize_intercept=False)
        lr = 1.0 / L

        # Fixed GD
        b_gd, info_gd, _ = gradient_descent(
            Xa, y, lr=lr, line_search=False, max_iter=max_iter_gd, tol=tol, reg=reg
        )

        # Backtracking GD
        b_bt, info_bt, _ = gradient_descent(
            Xa, y, lr=1.0, line_search=True, max_iter=max_iter_gd, tol=tol, reg=reg
        )

        # Damped Newton
        b_nw, info_nw, _ = newton_method(
            Xa, y, damping=True, max_iter=100, tol=tol, reg=reg
        )

        records.append({
            "correlation_rho": rho,
            "cond_X": cond_X,
            "cond_H0": cond_H0,
            "gd_fixed_iter": info_gd["iterations"],
            "gd_fixed_converged": info_gd["converged"],
            "gd_bt_iter": info_bt["iterations"],
            "gd_bt_converged": info_bt["converged"],
            "newton_iter": info_nw["iterations"],
            "newton_converged": info_nw["converged"],
            "newton_seconds": info_nw["seconds"],
            "gd_bt_seconds": info_bt["seconds"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "experiment_c_correlation.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.semilogy(df["correlation_rho"], df["cond_H0"], "o-", color="purple", lw=2, label=r"Hessian $\kappa(\nabla^2 J(0))$")
    ax1.set_xlabel(r"Feature Correlation $\rho$", fontsize=11)
    ax1.set_ylabel("Condition Number (log scale)", fontsize=11)
    ax1.set_title("Hessian Ill-Conditioning vs Correlation", fontsize=12)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    ax2.plot(df["correlation_rho"], df["newton_iter"], "s-", color="darkred", lw=2, label="Newton's Method")
    ax2.plot(df["correlation_rho"], df["gd_bt_iter"], "o-", color="steelblue", lw=2, label="GD (Armijo Backtracking)")
    ax2.set_xlabel(r"Feature Correlation $\rho$", fontsize=11)
    ax2.set_ylabel("Iterations to Convergence", fontsize=11)
    ax2.set_title("Iteration Count: 1st vs 2nd Order", fontsize=12)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "experiment_c_correlation.png", dpi=200)
    plt.close()

    return df
