# Optimization Methods for Logistic Regression

A from-first-principles computational research project comparing first- and second-order optimization algorithms for binary logistic regression.

## Key Highlights
- **Mathematical Derivations:** Bernoulli likelihood, negative log-likelihood (binary cross-entropy), analytical gradient, analytical Hessian, convexity proof, strict convexity conditions, non-existence of finite MLE on linearly separable data (Albert & Anderson 1984), and Lipschitz step-size bounds ($\eta < 2/L$).
- **Optimizers Implemented From Scratch (NumPy):**
  1. Batch Gradient Descent (Fixed Step Size)
  2. Gradient Descent with Armijo Backtracking Line Search
  3. Pure Newton's Method ($\alpha = 1$)
  4. Damped Newton's Method (Newton with Armijo Backtracking)
- **Mathematical Verification:** 40 automated unit tests verifying analytical gradient and Hessian against central finite differences to within machine-level relative precision ($< 10^{-8}$).
- **Controlled Numerical Experiments:**
  - **Experiment A:** Step-size sensitivity and validation of theoretical bound $\eta < 2/L$.
  - **Experiment B:** Feature scaling, condition numbers $\kappa(X)$, and Newton's affine invariance.
  - **Experiment C:** Feature correlation $\rho \in [0.0, 0.99]$ and Hessian ill-conditioning.
  - **Experiment D:** Dimensionality scaling ($p \in [10, 400]$) and $O(np)$ vs. $O(np^2 + p^3)$ complexity crossover.
  - **Experiment E:** Linear separability, parameter divergence $\|\beta\| \to \infty$, Hessian degeneration $W \to 0$, and L2 regularization.

## Repository Structure
```
.
├── pyproject.toml              # Project metadata & pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & instructions
├── logistic_optimizers.py      # Standalone runner for Wisconsin Breast Cancer benchmark
├── src/                        # Core optimization package
│   └── logistic_optimization/
│       ├── __init__.py
│       ├── objective.py        # Likelihood, gradient, Hessian, Lipschitz constant
│       ├── line_search.py      # Armijo backtracking line search
│       ├── gradient_descent.py # Fixed GD & Armijo backtracking GD
│       ├── newton.py           # Pure Newton & Damped Newton
│       ├── model.py            # Scikit-learn estimator interface
│       ├── data.py             # Synthetic data generators (kappa, rho, separability)
│       └── diagnostics.py      # Finite differences, condition numbers, metrics
├── experiments/                # Controlled research experiment scripts
│   ├── run_all.py              # Master runner for all experiments
│   ├── benchmark_breast_cancer.py
│   ├── learning_rate.py
│   ├── feature_scaling.py
│   ├── correlation.py
│   ├── dimensionality.py
│   └── separability.py
├── tests/                      # Automated unit test suite
│   ├── test_gradient.py        # Gradient finite-difference validation
│   ├── test_hessian.py         # Hessian finite-difference & PSD validation
│   └── test_optimizers.py      # Solution equivalence & quadratic convergence
├── results/                    # Generated benchmark tables, CSVs, and plots
└── report/                     # Comprehensive research reports
    ├── report.md               # Full Markdown research report
    └── report.tex              # LaTeX research paper
```

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Verification Suite (Unit Tests)
```bash
PYTHONPATH=src pytest tests/ -v
```

### 3. Run Full Research Experiment Pipeline
```bash
PYTHONPATH=.:src python experiments/run_all.py --out results
```

### 4. Run Wisconsin Breast Cancer Benchmark
```bash
python logistic_optimizers.py --out results
```

## Research Report
For full mathematical derivations, proofs, experimental curves, and analytical discussions, see [report/report.md](report/report.md).
