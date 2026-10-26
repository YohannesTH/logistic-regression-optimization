"""From-scratch optimization methods for binary logistic regression.

Run: python logistic_optimizers.py --out results
Dependencies: numpy, pandas, matplotlib, scikit-learn
The optimizers use only NumPy; scikit-learn supplies a public dataset and split/scaling utilities.
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

EPS = 1e-12

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def objective(X, y, beta, reg=1e-4):
    z = X @ beta
    penalty_mask = np.ones(beta.size)
    penalty_mask[0] = 0.0
    return float(np.mean(np.logaddexp(0.0, z) - y*z) + 0.5*reg*np.sum((penalty_mask*beta)**2))

def gradient(X, y, beta, reg=1e-4):
    p = sigmoid(X @ beta)
    penalty_mask = np.ones(beta.size)
    penalty_mask[0] = 0.0
    return X.T @ (p - y) / len(y) + reg * penalty_mask * beta

def hessian(X, y, beta, reg=1e-4):
    p = sigmoid(X @ beta)
    w = p * (1.0 - p)
    H = (X.T * w) @ X / len(y)
    penalty_mask = np.ones(beta.size)
    penalty_mask[0] = 0.0
    H += reg * np.diag(penalty_mask)
    return H

def gradient_descent(X, y, beta0=None, lr=0.1, max_iter=5000, tol=1e-8, reg=1e-4):
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, dtype=float).copy()
    history = [objective(X, y, beta, reg)]
    start = time.perf_counter()
    converged = False
    for it in range(1, max_iter + 1):
        g = gradient(X, y, beta, reg)
        if np.linalg.norm(g, ord=np.inf) < tol:
            converged = True
            break
        beta_new = beta - lr * g
        f_new = objective(X, y, beta_new, reg)
        if not np.isfinite(f_new):
            break
        beta = beta_new
        history.append(f_new)
    elapsed = time.perf_counter() - start
    final_g = gradient(X, y, beta, reg)
    final_g_inf = float(np.linalg.norm(final_g, ord=np.inf))
    return beta, {
        'method': 'Gradient descent',
        'iterations': len(history) - 1,
        'objective': history[-1],
        'gradient_inf_norm': final_g_inf,
        'converged': bool(converged or final_g_inf < tol),
        'seconds': elapsed
    }, history

def newton_method(X, y, beta0=None, max_iter=100, tol=1e-8, reg=1e-4, armijo=1e-4):
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, dtype=float).copy()
    history = [objective(X, y, beta, reg)]
    start = time.perf_counter()
    converged = False
    for it in range(1, max_iter + 1):
        g = gradient(X, y, beta, reg)
        if np.linalg.norm(g, ord=np.inf) < tol:
            converged = True
            break
        H = hessian(X, y, beta, reg)
        try:
            direction = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(H, g, rcond=None)[0]
        decrement = float(g @ direction)
        if not np.isfinite(decrement) or decrement <= 0:
            direction = g
            decrement = float(g @ g)
        f0 = history[-1]
        step = 1.0
        while step > 1e-12:
            candidate = beta - step * direction
            fc = objective(X, y, candidate, reg)
            if np.isfinite(fc) and fc <= f0 - armijo * step * decrement:
                break
            step *= 0.5
        if step <= 1e-12:
            break
        beta = candidate
        history.append(fc)
    elapsed = time.perf_counter() - start
    final_g = gradient(X, y, beta, reg)
    final_g_inf = float(np.linalg.norm(final_g, ord=np.inf))
    return beta, {
        'method': 'Newton (backtracking)',
        'iterations': len(history) - 1,
        'objective': history[-1],
        'gradient_inf_norm': final_g_inf,
        'converged': bool(converged or final_g_inf < tol),
        'seconds': elapsed
    }, history

def augment(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def evaluate(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, prob),
        'test_log_loss': log_loss(y, np.clip(prob, EPS, 1.0 - EPS))
    }

def main(out_dir='results'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    Xraw, y = data.data, data.target.astype(float)
    Xtr0, Xte0, ytr, yte = train_test_split(Xraw, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr0)
    Xte = scaler.transform(Xte0)
    Xa, Xb = augment(Xtr), augment(Xte)

    summaries = []
    histories = {}
    for name, solver in [('Gradient descent', gradient_descent), ('Newton (backtracking)', newton_method)]:
        beta, info, hist = (solver(Xa, ytr, lr=3.0, max_iter=35000) if name == 'Gradient descent' else solver(Xa, ytr))
        info.update(evaluate(yte, sigmoid(Xb @ beta)))
        summaries.append(info)
        histories[name] = hist
    pd.DataFrame(summaries).to_csv(out / 'benchmark.csv', index=False)

    # Convergence plot
    plt.figure()
    for name, hist in histories.items():
        plt.semilogy(np.arange(len(hist)), np.maximum(np.array(hist) - min(min(v) for v in histories.values()) + 1e-15, 1e-15), label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Objective gap (shifted, log scale)')
    plt.title('Convergence on standardized breast-cancer data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'convergence.png', dpi=180)
    plt.close()

    # Learning-rate sensitivity
    lr_rows = []
    for lr in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
        beta, info, hist = gradient_descent(Xa, ytr, lr=lr, max_iter=3000, tol=1e-7)
        lr_rows.append({
            'learning_rate': lr,
            **{k: info[k] for k in ['iterations', 'objective', 'gradient_inf_norm', 'converged', 'seconds']},
            **evaluate(yte, sigmoid(Xb @ beta))
        })
    pd.DataFrame(lr_rows).to_csv(out / 'learning_rate_sensitivity.csv', index=False)

    # Scaling comparison
    scale_rows = []
    for label, A, B in [('Raw features', augment(Xtr0), augment(Xte0)), ('Standardized features', Xa, Xb)]:
        beta, info, hist = gradient_descent(A, ytr, lr=0.01, max_iter=3000, tol=1e-7)
        scale_rows.append({
            'features': label,
            **{k: info[k] for k in ['iterations', 'objective', 'gradient_inf_norm', 'converged', 'seconds']},
            **evaluate(yte, sigmoid(B @ beta))
        })
    pd.DataFrame(scale_rows).to_csv(out / 'scaling_comparison.csv', index=False)

    # Initialization sensitivity
    init_rows = []
    rng = np.random.default_rng(7)
    for label, b0 in [('Zero initialization', np.zeros(Xa.shape[1])), ('Small random initialization', rng.normal(0, 0.1, Xa.shape[1]))]:
        for name, solver in [('Gradient descent', gradient_descent), ('Newton (backtracking)', newton_method)]:
            beta, info, hist = solver(Xa, ytr, beta0=b0)
            init_rows.append({'initialization': label, **info, **evaluate(yte, sigmoid(Xb @ beta))})
    pd.DataFrame(init_rows).to_csv(out / 'initialization_comparison.csv', index=False)

    # Dimensionality
    dim_rows = []
    for k in [2, 5, 10, 20, 30]:
        A = augment(Xtr[:, :k])
        B = augment(Xte[:, :k])
        for name, solver in [('Gradient descent', gradient_descent), ('Newton (backtracking)', newton_method)]:
            beta, info, hist = solver(A, ytr, max_iter=3000 if name == 'Gradient descent' else 100)
            dim_rows.append({'n_features': k, **info, **evaluate(yte, sigmoid(B @ beta))})
    pd.DataFrame(dim_rows).to_csv(out / 'dimensionality_comparison.csv', index=False)

    metadata = {
        'dataset': 'Wisconsin Diagnostic Breast Cancer (scikit-learn)',
        'n_samples': int(len(y)),
        'n_features': int(Xraw.shape[1]),
        'train_samples': int(len(ytr)),
        'test_samples': int(len(yte)),
        'test_size': 0.25,
        'random_state': 42,
        'regularization_l2': 1e-4,
        'regularization_excludes_intercept': True,
        'note': 'Metrics are on held-out test data. Timings are environment-dependent; dimensionality study uses the first k dataset columns, not selected best features.'
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Benchmark:')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('\nLearning-rate sensitivity:')
    print(pd.DataFrame(lr_rows)[['learning_rate', 'iterations', 'objective', 'converged', 'accuracy']].to_string(index=False))
    print('\nScaling comparison:')
    print(pd.DataFrame(scale_rows)[['features', 'iterations', 'objective', 'converged', 'accuracy']].to_string(index=False))
    print(f'\nSaved results to: {out.resolve()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    main(args.out)
