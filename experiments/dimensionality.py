"""Experiment D: Dimensionality Scaling and Computational Complexity.

Investigates:
1. Scaling of per-iteration cost: O(np) for GD vs O(np^2 + p^3) for Newton.
2. Controlled variation of dimension p in [10, 50, 100, 200, 400] with fixed n=1000.
3. Total wall-clock time vs iteration trade-off.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_optimization.gradient_descent import gradient_descent
from logistic_optimization.newton import newton_method
from logistic_optimization.data import generate_ill_conditioned_data


def run_experiment_dimensionality(out_dir: Path) -> pd.DataFrame:
    n_samples = 1000
    p_values = [10, 30, 60, 120, 250, 400]
    records = []

    tol = 1e-6
    reg = 1e-4

    for p in p_values:
        X, y, _ = generate_ill_conditioned_data(
            n_samples=n_samples, n_features=p, condition_number=20.0, random_state=42 + p
        )
        Xa = np.column_stack([np.ones(n_samples), X])

        # Benchmark GD (Armijo backtracking)
        # Average over 3 runs for reliable timing
        times_gd = []
        for _ in range(3):
            t0 = time.perf_counter()
            _, info_gd, _ = gradient_descent(
                Xa, y, lr=1.0, line_search=True, max_iter=2000, tol=tol, reg=reg
            )
            times_gd.append(time.perf_counter() - t0)
        gd_time = float(np.mean(times_gd))
        gd_iter = info_gd["iterations"]
        gd_iter_time = gd_time / max(1, gd_iter)

        # Benchmark Damped Newton
        times_nw = []
        for _ in range(3):
            t0 = time.perf_counter()
            _, info_nw, _ = newton_method(
                Xa, y, damping=True, max_iter=50, tol=tol, reg=reg
            )
            times_nw.append(time.perf_counter() - t0)
        nw_time = float(np.mean(times_nw))
        nw_iter = info_nw["iterations"]
        nw_iter_time = nw_time / max(1, nw_iter)

        records.append({
            "n_features": p,
            "n_samples": n_samples,
            "gd_iterations": gd_iter,
            "gd_total_seconds": gd_time,
            "gd_per_iter_ms": gd_iter_time * 1000.0,
            "newton_iterations": nw_iter,
            "newton_total_seconds": nw_time,
            "newton_per_iter_ms": nw_iter_time * 1000.0,
            "speedup_ratio": gd_time / max(1e-6, nw_time),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "experiment_d_dimensionality.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.loglog(df["n_features"], df["gd_per_iter_ms"], "o-", color="steelblue", lw=2, label="GD per iteration: O(np)")
    ax1.loglog(df["n_features"], df["newton_per_iter_ms"], "s-", color="darkred", lw=2, label="Newton per iteration: O(np² + p³)")
    ax1.set_xlabel("Number of Features p (log scale)", fontsize=11)
    ax1.set_ylabel("Time per Iteration (ms, log scale)", fontsize=11)
    ax1.set_title("Computational Cost per Iteration vs Dimension", fontsize=12)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    ax2.plot(df["n_features"], df["gd_total_seconds"], "o-", color="steelblue", lw=2, label="GD Total Time")
    ax2.plot(df["n_features"], df["newton_total_seconds"], "s-", color="darkred", lw=2, label="Newton Total Time")
    ax2.set_xlabel("Number of Features p", fontsize=11)
    ax2.set_ylabel("Total Wall-Clock Time (s)", fontsize=11)
    ax2.set_title("Total Convergence Time vs Dimension", fontsize=12)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "experiment_d_dimensionality.png", dpi=200)
    plt.close()

    return df
