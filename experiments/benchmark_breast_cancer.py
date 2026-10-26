"""Real-Dataset Benchmark: Wisconsin Diagnostic Breast Cancer.

Compares all 4 optimization methods:
1. Gradient Descent (Fixed Step Size)
2. Gradient Descent (Armijo Backtracking Line Search)
3. Pure Newton's Method (Unit Step Size)
4. Damped Newton's Method (Armijo Backtracking)

Averages runtimes over repeated trials and logs complete optimization and classification metrics.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.objective import sigmoid, lipschitz_constant
from logistic_optimization.diagnostics import evaluate_classification, compute_condition_number


def run_benchmark_breast_cancer(out_dir: Path) -> pd.DataFrame:
    data = load_breast_cancer()
    Xraw, y = data.data, data.target.astype(float)
    Xtr0, Xte0, ytr, yte = train_test_split(Xraw, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr0)
    Xte = scaler.transform(Xte0)

    Xa = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    Xb = np.column_stack([np.ones(Xte.shape[0]), Xte])

    reg = 1e-4
    tol = 1e-8
    L = lipschitz_constant(Xa, reg=reg, penalize_intercept=False)
    eta_theoretical = 1.0 / L

    solvers = [
        ("Gradient Descent (Fixed eta=3.0)", lambda: gradient_descent(Xa, ytr, lr=3.0, line_search=False, max_iter=35000, tol=tol, reg=reg)),
        ("Gradient Descent (Armijo Backtracking)", lambda: gradient_descent(Xa, ytr, lr=3.0, line_search=True, max_iter=15000, tol=tol, reg=reg)),
        ("Pure Newton (Unit Step alpha=1)", lambda: newton_method(Xa, ytr, damping=False, max_iter=50, tol=tol, reg=reg)),
        ("Damped Newton (Armijo Backtracking)", lambda: newton_method(Xa, ytr, damping=True, max_iter=50, tol=tol, reg=reg)),
    ]

    summaries = []
    histories = {}
    N_REPEATS = 10

    for name, run_fn in solvers:
        # Run once to get final solution and history
        beta, info, hist = run_fn()
        histories[name] = hist

        # Average wall-clock runtime over repeated trials
        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            run_fn()
            times.append(time.perf_counter() - t0)
        avg_time = float(np.mean(times))
        std_time = float(np.std(times))

        metrics = evaluate_classification(yte, sigmoid(Xb @ beta))

        row = {
            "method": name,
            "iterations": info["iterations"],
            "final_objective": info["objective"],
            "gradient_inf_norm": info["gradient_inf_norm"],
            "gradient_2_norm": info["gradient_2_norm"],
            "converged": info["converged"],
            "seconds_mean": avg_time,
            "seconds_std": std_time,
            "n_fev": info["n_fev"],
            "n_gev": info["n_gev"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "test_log_loss": metrics["log_loss"],
        }
        summaries.append(row)

    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "benchmark.csv", index=False)

    # Convergence plot
    plt.figure(figsize=(9, 5))
    min_obj = min(min(h) for h in histories.values() if len(h) > 0 and np.isfinite(min(h)))
    for name, hist in histories.items():
        gap = np.maximum(np.array(hist) - min_obj + 1e-15, 1e-15)
        plt.semilogy(np.arange(len(gap)), gap, label=name, lw=1.8)

    plt.xlabel("Iteration k", fontsize=11)
    plt.ylabel(r"Objective Gap $J(\beta_k) - J^*$ (log scale)", fontsize=11)
    plt.title("Convergence Comparison on Wisconsin Breast Cancer Data", fontsize=12)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=200)
    plt.close()

    # Metadata
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "n_samples": int(len(y)),
        "n_features": int(Xraw.shape[1]),
        "train_samples": int(len(ytr)),
        "test_samples": int(len(yte)),
        "test_size": 0.25,
        "random_state": 42,
        "regularization_l2": reg,
        "gradient_lipschitz_L": L,
        "theoretical_max_step_size": 2.0 / L,
        "n_timing_repeats": N_REPEATS,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return df
