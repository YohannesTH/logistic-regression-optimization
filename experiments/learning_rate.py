"""Experiment A: Learning Rate Sensitivity and Armijo Backtracking Line Search.

Investigates:
1. Convergence behavior across fixed learning rates {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0}.
2. Comparison with adaptive Armijo backtracking line search.
3. Analysis of theoretical step-size bound eta < 2 / L based on Lipschitz constant L.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.objective import lipschitz_constant
from logistic_optimization.diagnostics import evaluate_classification


def run_experiment_learning_rate(
    Xa: np.ndarray,
    ytr: np.ndarray,
    Xb: np.ndarray,
    yte: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    L = lipschitz_constant(Xa, reg=1e-4, penalize_intercept=False)
    eta_max = 2.0 / L
    eta_recommended = 1.0 / L

    rates = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    records = []
    histories: Dict[str, List[float]] = {}

    max_iter = 4000
    tol = 1e-7
    reg = 1e-4

    for lr in rates:
        beta, info, hist = gradient_descent(
            Xa, ytr, lr=lr, line_search=False, max_iter=max_iter, tol=tol, reg=reg
        )
        from logistic_optimization.objective import sigmoid
        preds = evaluate_classification(yte, sigmoid(Xb @ beta))
        label = f"GD (eta={lr})"
        histories[label] = hist
        records.append({
            "method": label,
            "learning_rate": lr,
            "type": "fixed",
            "iterations": info["iterations"],
            "final_objective": info["objective"],
            "gradient_inf_norm": info["gradient_inf_norm"],
            "converged": info["converged"],
            "seconds": info["seconds"],
            "n_fev": info["n_fev"],
            "n_gev": info["n_gev"],
            "test_accuracy": preds["accuracy"],
            "test_log_loss": preds["log_loss"],
        })

    # Armijo Backtracking
    beta_bt, info_bt, hist_bt = gradient_descent(
        Xa, ytr, lr=1.0, line_search=True, max_iter=max_iter, tol=tol, reg=reg
    )
    from logistic_optimization.objective import sigmoid
    preds_bt = evaluate_classification(yte, sigmoid(Xb @ beta_bt))
    label_bt = "GD (Armijo Backtracking)"
    histories[label_bt] = hist_bt
    records.append({
        "method": label_bt,
        "learning_rate": "adaptive",
        "type": "backtracking",
        "iterations": info_bt["iterations"],
        "final_objective": info_bt["objective"],
        "gradient_inf_norm": info_bt["gradient_inf_norm"],
        "converged": info_bt["converged"],
        "seconds": info_bt["seconds"],
        "n_fev": info_bt["n_fev"],
        "n_gev": info_bt["n_gev"],
        "test_accuracy": preds_bt["accuracy"],
        "test_log_loss": preds_bt["log_loss"],
    })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "experiment_a_learning_rate.csv", index=False)

    # Plot trajectories
    plt.figure(figsize=(9, 5))
    min_obj = min(min(h) for h in histories.values() if len(h) > 0 and np.isfinite(min(h)))
    for name, h in histories.items():
        gap = np.maximum(np.array(h) - min_obj + 1e-15, 1e-15)
        plt.semilogy(np.arange(len(gap)), gap, label=name, lw=1.5)

    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Iteration k", fontsize=11)
    plt.ylabel(r"Objective Gap $J(\beta_k) - J^*$ (log scale)", fontsize=11)
    plt.title(f"Experiment A: Step-Size Sensitivity (Theoretical Bound $2/L \\approx {eta_max:.2f}$)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "experiment_a_learning_rate.png", dpi=200)
    plt.close()

    return df
