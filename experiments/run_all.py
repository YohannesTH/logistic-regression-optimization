"""Master Experiment Runner: Executes full research pipeline and generates all results and figures.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.benchmark_breast_cancer import run_benchmark_breast_cancer
from experiments.learning_rate import run_experiment_learning_rate
from experiments.feature_scaling import run_experiment_scaling
from experiments.correlation import run_experiment_correlation
from experiments.dimensionality import run_experiment_dimensionality
from experiments.separability import run_experiment_separability


def main(out_dir_str: str = "results") -> None:
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Optimization Research Pipeline -> Output: {out_dir.resolve()} ===\n")

    # 1. Real-world benchmark (Wisconsin Breast Cancer)
    print("1. Running Primary Benchmark (Wisconsin Breast Cancer)...")
    df_bench = run_benchmark_breast_cancer(out_dir)
    print(df_bench[["method", "iterations", "final_objective", "converged", "seconds_mean", "accuracy"]].to_string(index=False))

    # Prepare data for experiments A and B
    data = load_breast_cancer()
    Xraw, y = data.data, data.target.astype(float)
    Xtr0, Xte0, ytr, yte = train_test_split(Xraw, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr0)
    Xte = scaler.transform(Xte0)
    Xa = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    Xb = np.column_stack([np.ones(Xte.shape[0]), Xte])

    # 2. Experiment A: Learning Rate & Backtracking
    print("\n2. Running Experiment A: Learning Rate Sensitivity & Line Search...")
    df_lr = run_experiment_learning_rate(Xa, ytr, Xb, yte, out_dir)
    print(df_lr[["method", "iterations", "final_objective", "converged", "seconds", "test_accuracy"]].to_string(index=False))

    # 3. Experiment B: Feature Scaling & Condition Numbers
    print("\n3. Running Experiment B: Feature Scaling & Condition Numbers...")
    df_scale = run_experiment_scaling(Xtr0, Xte0, Xtr, Xte, ytr, yte, out_dir)
    print(df_scale[["features", "condition_number", "method", "iterations", "converged", "accuracy"]].to_string(index=False))

    # 4. Experiment C: Feature Correlation (Synthetic)
    print("\n4. Running Experiment C: Feature Correlation & Hessian Ill-Conditioning...")
    df_corr = run_experiment_correlation(out_dir)
    print(df_corr[["correlation_rho", "cond_H0", "gd_bt_iter", "newton_iter", "newton_seconds"]].to_string(index=False))

    # 5. Experiment D: Dimensionality Scaling (Synthetic)
    print("\n5. Running Experiment D: Dimensionality Scaling...")
    df_dim = run_experiment_dimensionality(out_dir)
    print(df_dim[["n_features", "gd_iterations", "gd_per_iter_ms", "newton_iterations", "newton_per_iter_ms", "speedup_ratio"]].to_string(index=False))

    # 6. Experiment E: Linear Separability & Non-existence of MLE
    print("\n6. Running Experiment E: Linear Separability & MLE Existence...")
    df_sep = run_experiment_separability(out_dir)
    print(df_sep.to_string(index=False))

    print(f"\n=== All experiments completed successfully! Results written to {out_dir.resolve()} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all optimization experiments")
    parser.add_argument("--out", default="results", help="Directory to save artifacts and plots")
    args = parser.parse_args()
    main(args.out)
