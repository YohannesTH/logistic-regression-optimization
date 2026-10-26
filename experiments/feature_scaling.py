"""Experiment B: Feature Scaling and Condition Number Effects.

Investigates:
1. Ill-conditioning introduced by unscaled features with disparate magnitudes.
2. Comparison of GD (fixed and backtracking) vs Newton on raw and standardized features.
3. Affine invariance of Newton's method vs GD's sensitivity to scaling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.diagnostics import compute_condition_number, evaluate_classification
from logistic_optimization.objective import sigmoid


def run_experiment_scaling(
    Xtr_raw: np.ndarray,
    Xte_raw: np.ndarray,
    Xtr_std: np.ndarray,
    Xte_std: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    # Augment with intercept
    Xa_raw = np.column_stack([np.ones(Xtr_raw.shape[0]), Xtr_raw])
    Xb_raw = np.column_stack([np.ones(Xte_raw.shape[0]), Xte_raw])
    Xa_std = np.column_stack([np.ones(Xtr_std.shape[0]), Xtr_std])
    Xb_std = np.column_stack([np.ones(Xte_std.shape[0]), Xte_std])

    cond_raw = compute_condition_number(Xa_raw)
    cond_std = compute_condition_number(Xa_std)

    records = []
    histories: Dict[str, List[float]] = {}

    configs = [
        ("Raw features", Xa_raw, Xb_raw, cond_raw),
        ("Standardized features", Xa_std, Xb_std, cond_std),
    ]

    for label, A, B, cond_val in configs:
        # Fixed GD (lr=0.01)
        b_gd, info_gd, h_gd = gradient_descent(A, ytr, lr=0.01, line_search=False, max_iter=3000, tol=1e-7)
        p_gd = evaluate_classification(yte, sigmoid(B @ b_gd))
        histories[f"{label} - GD (lr=0.01)"] = h_gd
        records.append({
            "features": label,
            "condition_number": cond_val,
            "method": "GD (lr=0.01)",
            "iterations": info_gd["iterations"],
            "final_objective": info_gd["objective"],
            "gradient_inf_norm": info_gd["gradient_inf_norm"],
            "converged": info_gd["converged"],
            "seconds": info_gd["seconds"],
            "accuracy": p_gd["accuracy"],
            "log_loss": p_gd["log_loss"],
        })

        # Backtracking GD
        b_bt, info_bt, h_bt = gradient_descent(A, ytr, lr=1.0, line_search=True, max_iter=3000, tol=1e-7)
        p_bt = evaluate_classification(yte, sigmoid(B @ b_bt))
        histories[f"{label} - GD (Armijo)"] = h_bt
        records.append({
            "features": label,
            "condition_number": cond_val,
            "method": "GD (Armijo)",
            "iterations": info_bt["iterations"],
            "final_objective": info_bt["objective"],
            "gradient_inf_norm": info_bt["gradient_inf_norm"],
            "converged": info_bt["converged"],
            "seconds": info_bt["seconds"],
            "accuracy": p_bt["accuracy"],
            "log_loss": p_bt["log_loss"],
        })

        # Damped Newton
        b_newt, info_newt, h_newt = newton_method(A, ytr, damping=True, max_iter=100, tol=1e-7)
        p_newt = evaluate_classification(yte, sigmoid(B @ b_newt))
        histories[f"{label} - Newton (Armijo)"] = h_newt
        records.append({
            "features": label,
            "condition_number": cond_val,
            "method": "Newton (Armijo)",
            "iterations": info_newt["iterations"],
            "final_objective": info_newt["objective"],
            "gradient_inf_norm": info_newt["gradient_inf_norm"],
            "converged": info_newt["converged"],
            "seconds": info_newt["seconds"],
            "accuracy": p_newt["accuracy"],
            "log_loss": p_newt["log_loss"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "experiment_b_feature_scaling.csv", index=False)

    # Plot
    plt.figure(figsize=(9, 5))
    min_obj = min(min(h) for h in histories.values() if len(h) > 0 and np.isfinite(min(h)))
    for name, h in histories.items():
        gap = np.maximum(np.array(h) - min_obj + 1e-15, 1e-15)
        plt.semilogy(np.arange(len(gap)), gap, label=name, lw=1.5)

    plt.xlabel("Iteration k", fontsize=11)
    plt.ylabel(r"Objective Gap $J(\beta_k) - J^*$ (log scale)", fontsize=11)
    plt.title(f"Experiment B: Scaling Impact (Raw $\\kappa={cond_raw:.1e}$ vs Std $\\kappa={cond_std:.1e}$)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "experiment_b_feature_scaling.png", dpi=200)
    plt.close()

    return df
