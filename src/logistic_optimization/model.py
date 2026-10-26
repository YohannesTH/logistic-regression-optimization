"""Object-Oriented Logistic Regression Estimator Interface.

Compatible with scikit-learn conventions: fit, predict_proba, predict, score.
Supports all 4 optimization solvers:
1. 'gd'               - Fixed step Gradient Descent
2. 'gd_backtracking'  - Gradient Descent with Armijo Line Search
3. 'newton_pure'      - Pure Newton's Method (unit step)
4. 'newton_damping'   - Damped Newton's Method (Newton + Armijo)
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

from .objective import sigmoid
from .gradient_descent import gradient_descent
from .newton import newton_method
from .diagnostics import evaluate_classification


class LogisticRegressionModel:
    def __init__(
        self,
        solver: str = "newton_damping",
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-7,
        reg: float = 1e-4,
        fit_intercept: bool = True,
        c1: float = 1e-4,
        rho: float = 0.5,
    ):
        self.solver = solver.lower()
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.fit_intercept = fit_intercept
        self.c1 = c1
        self.rho = rho

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.beta_: np.ndarray | None = None
        self.info_: Dict[str, Any] = {}
        self.history_: List[float] = []

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, beta0: np.ndarray | None = None) -> "LogisticRegressionModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xa = self._augment(X)

        if self.solver in ["gd", "gradient_descent"]:
            beta, info, hist = gradient_descent(
                Xa, y, beta0=beta0, lr=self.lr, line_search=False,
                max_iter=self.max_iter, tol=self.tol, reg=self.reg,
                penalize_intercept=False if self.fit_intercept else True,
            )
        elif self.solver in ["gd_backtracking", "gd_armijo"]:
            beta, info, hist = gradient_descent(
                Xa, y, beta0=beta0, lr=self.lr, line_search=True,
                max_iter=self.max_iter, tol=self.tol, reg=self.reg,
                penalize_intercept=False if self.fit_intercept else True,
                c1=self.c1, rho=self.rho,
            )
        elif self.solver in ["newton_pure", "pure_newton"]:
            beta, info, hist = newton_method(
                Xa, y, beta0=beta0, damping=False,
                max_iter=self.max_iter, tol=self.tol, reg=self.reg,
                penalize_intercept=False if self.fit_intercept else True,
            )
        elif self.solver in ["newton_damping", "damped_newton", "newton"]:
            beta, info, hist = newton_method(
                Xa, y, beta0=beta0, damping=True,
                max_iter=self.max_iter, tol=self.tol, reg=self.reg,
                penalize_intercept=False if self.fit_intercept else True,
                c1=self.c1, rho=self.rho,
            )
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        self.beta_ = beta
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.info_ = info
        self.history_ = hist
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model is not fitted yet.")
        Xa = self._augment(np.asarray(X, dtype=float))
        p1 = sigmoid(Xa @ self.beta_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        p1 = self.predict_proba(X)[:, 1]
        return evaluate_classification(y, p1)
