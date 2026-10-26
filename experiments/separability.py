"""Experiment E: Linear Separability, Non-Existence of Finite MLE, and L2 Regularization.

Investigates:
1. The behavior of unregularized logistic regression (lambda=0) on linearly separable data.
2. Divergence of parameter norm ||beta_k|| -> inf and degeneration of Hessian curvature W -> 0.
3. How L2 regularization restores strong convexity and guarantees a finite unique minimizer.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_optimization.data import generate_separable_data
from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.objective import gradient, hessian


def run_experiment_separability(out_dir: Path) -> pd.DataFrame:
    X, y, _ = generate_separable_data(n_samples=250, n_features=4, margin=0.5, random_state=42)
    Xa = np.column_stack([np.ones(X.shape[0]), X])

    lambdas = [0.0, 1e-4, 1e-2, 1e-1]
    records = []
    trajectories = {}

    for reg in lambdas:
        # Run damped Newton to observe iterate norm growth and Hessian condition
        beta = np.zeros(Xa.shape[1])
        norm_history = [float(np.linalg.norm(beta))]
        obj_history = []
        min_hessian_eig = []

        # Custom loop to record norm at each Newton iteration
        for it in range(40):
            g = gradient(Xa, y, beta, reg=reg, penalize_intercept=True)
            H = hessian(Xa, y, beta, reg=reg, penalize_intercept=True)
            eigs = np.linalg.eigvalsh(H)
            min_hessian_eig.append(float(np.min(eigs)))

            try:
                s = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                s = np.linalg.lstsq(H, g, rcond=None)[0]

            # Step with small damping for stability
            step = 1.0
            beta_next = beta - step * s
            beta = beta_next
            norm_history.append(float(np.linalg.norm(beta)))

        label = f"lambda={reg}"
        trajectories[label] = {
            "norms": norm_history,
            "min_eigs": min_hessian_eig,
        }

        records.append({
            "regularization_lambda": reg,
            "final_beta_norm": norm_history[-1],
            "min_hessian_eigenvalue": min_hessian_eig[-1],
            "finite_mle_guaranteed": bool(reg > 0),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "experiment_e_separability.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, traj in trajectories.items():
        ax1.plot(traj["norms"], lw=2, label=label)
    ax1.set_xlabel("Newton Iteration k", fontsize=11)
    ax1.set_ylabel(r"Coefficient Norm $\|\beta_k\|_2$", fontsize=11)
    ax1.set_title("Coefficient Norm Growth on Separable Data", fontsize=12)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend()

    for label, traj in trajectories.items():
        ax2.semilogy(np.maximum(traj["min_eigs"], 1e-16), lw=2, label=label)
    ax2.set_xlabel("Newton Iteration k", fontsize=11)
    ax2.set_ylabel(r"Minimum Hessian Eigenvalue $\lambda_{\min}(H)$ (log scale)", fontsize=11)
    ax2.set_title("Hessian Curvature Degeneration ($W \to 0$)", fontsize=12)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "experiment_e_separability.png", dpi=200)
    plt.close()

    return df
