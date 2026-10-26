"""Optimization Methods for Logistic Regression.

Implements from-scratch first- and second-order optimization algorithms,
objective functions, line search strategies, and diagnostic utilities.
"""

from .objective import sigmoid, objective, gradient, hessian, lipschitz_constant
from .line_search import armijo_backtracking
from .gradient_descent import gradient_descent
from .newton import newton_method
from .model import LogisticRegressionModel
from .diagnostics import compute_condition_number, evaluate_classification, check_gradient, check_hessian

__all__ = [
    "sigmoid",
    "objective",
    "gradient",
    "hessian",
    "lipschitz_constant",
    "armijo_backtracking",
    "gradient_descent",
    "newton_method",
    "LogisticRegressionModel",
    "compute_condition_number",
    "evaluate_classification",
    "check_gradient",
    "check_hessian",
]
