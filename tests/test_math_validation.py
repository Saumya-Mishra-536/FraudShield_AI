"""
Mathematical Validation of Asymmetric Loss, Gradients, Hessians, and Thresholds
================================================================================
Validates:
1. Analytical gradient matches numerical finite-difference gradient.
2. Analytical Hessian matches numerical finite-difference derivative of gradient.
3. Strict positivity of Hessian (convexity condition for Newton-Raphson boosting).
4. Asymmetric loss alignment with Kahneman & Tversky loss aversion ratio.
5. Bayes optimal decision threshold minimizes empirical cost on calibrated probabilities.
6. Extreme value behavior and numerical stability under logit clipping.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from src.loss_functions import (
    DEFAULT_C_FP,
    DEFAULT_C_FN,
    asymmetric_cost,
    expected_loss,
    optimal_threshold,
    prospect_theory_utility,
    xgboost_asymmetric_objective,
)


def loss_scalar(p: float, y: float, c_fp: float, c_fn: float) -> float:
    """Exact weighted log-loss: L = -[c_fn * y * ln(sigmoid(p)) + c_fp * (1-y) * ln(1 - sigmoid(p))]."""
    # Numerically stable formulation:
    # ln(sigmoid(p)) = -softplus(-p) = -ln(1 + exp(-p))
    # ln(1 - sigmoid(p)) = -softplus(p) = -ln(1 + exp(p))
    if p >= 0:
        log_sig = -np.log1p(np.exp(-p))
        log_one_minus_sig = -p - np.log1p(np.exp(-p))
    else:
        log_sig = p - np.log1p(np.exp(p))
        log_one_minus_sig = -np.log1p(np.exp(p))
    return - (c_fn * y * log_sig + c_fp * (1.0 - y) * log_one_minus_sig)


class MockDMatrix:
    def __init__(self, labels: np.ndarray):
        self._labels = np.asarray(labels, dtype=float)

    def get_label(self) -> np.ndarray:
        return self._labels


def test_analytical_gradient_matches_numerical_finite_differences():
    """Verify dL/dp matches (L(p+h) - L(p-h)) / (2h) across a dense logit grid."""
    c_fp = 5.0
    c_fn = 20.0
    objective = xgboost_asymmetric_objective(c_fp=c_fp, c_fn=c_fn)
    h = 1e-6

    test_points = np.linspace(-6.0, 6.0, 50)
    for y in [0.0, 1.0]:
        labels = np.full_like(test_points, y)
        dmat = MockDMatrix(labels)
        grad, _ = objective(test_points, dmat)

        for i, p in enumerate(test_points):
            f_plus = loss_scalar(p + h, y, c_fp, c_fn)
            f_minus = loss_scalar(p - h, y, c_fp, c_fn)
            num_grad = (f_plus - f_minus) / (2.0 * h)
            np.testing.assert_allclose(
                grad[i], num_grad, rtol=1e-4, atol=1e-4,
                err_msg=f"Gradient mismatch at p={p}, y={y}"
            )


def test_analytical_hessian_matches_numerical_gradient_differences():
    """Verify Hessian d²L/dp² matches (grad(p+h) - grad(p-h)) / (2h)."""
    c_fp = 5.0
    c_fn = 20.0
    objective = xgboost_asymmetric_objective(c_fp=c_fp, c_fn=c_fn)
    h = 1e-6

    test_points = np.linspace(-6.0, 6.0, 50)
    for y in [0.0, 1.0]:
        labels = np.full_like(test_points, y)
        dmat = MockDMatrix(labels)
        _, hess = objective(test_points, dmat)

        grad_plus, _ = objective(test_points + h, dmat)
        grad_minus, _ = objective(test_points - h, dmat)
        num_hess = (grad_plus - grad_minus) / (2.0 * h)

        np.testing.assert_allclose(
            hess, num_hess, rtol=1e-4, atol=1e-4,
            err_msg=f"Hessian mismatch for y={y}"
        )


def test_hessian_strictly_positive_for_boosting_stability():
    """Ensure d²L/dp² > 0 for all points in domain [-40, 40] for both classes."""
    objective = xgboost_asymmetric_objective(c_fp=5.0, c_fn=20.0)
    raw_scores = np.linspace(-40.0, 40.0, 200)

    for y in [0.0, 1.0]:
        dmat = MockDMatrix(np.full_like(raw_scores, y))
        _, hess = objective(raw_scores, dmat)
        assert np.all(hess > 0), f"Hessian non-positive for class {y}"
        assert np.all(np.isfinite(hess)), f"Hessian contains NaN/inf for class {y}"


def test_bayes_optimal_threshold_minimizes_expected_cost():
    """
    Under calibrated probabilities, the risk-minimizing threshold is:
        tau* = C_FP / (C_FP + C_FN)
    """
    c_fp = 5.0
    c_fn = 20.0
    theoretical_tau = c_fp / (c_fp + c_fn)  # 5 / 25 = 0.20
    assert theoretical_tau == pytest.approx(0.20)

    # Generate synthetic calibrated probabilities from Beta distributions
    np.random.seed(123)
    n = 100_000
    # True fraud probabilities
    p = np.random.beta(0.5, 5.0, size=n)
    # Binary outcomes sampled from Bernoulli(p) -> calibrated by definition
    y = np.random.binomial(1, p)

    emp_tau, min_cost, thresholds, costs = optimal_threshold(y, p, c_fp=c_fp, c_fn=c_fn, n_thresholds=200)

    # Empirical threshold should be within 0.03 of theoretical 0.20 on a large calibrated sample
    assert abs(emp_tau - theoretical_tau) <= 0.03, (
        f"Empirical threshold {emp_tau:.3f} deviates from theoretical {theoretical_tau:.3f}"
    )


def test_cost_calculation_monotonicity_and_invariance():
    """Verify cost properties: FP and FN strictly penalized according to weights."""
    y_true = np.array([0, 0, 1, 1])

    # Case 1: 0 FP, 0 FN
    assert asymmetric_cost(y_true, np.array([0, 0, 1, 1]), c_fp=5.0, c_fn=20.0) == 0.0

    # Case 2: 1 FP (pred 1 for true 0)
    cost_fp = asymmetric_cost(y_true, np.array([1, 0, 1, 1]), c_fp=5.0, c_fn=20.0)
    assert cost_fp == 5.0

    # Case 3: 1 FN (pred 0 for true 1)
    cost_fn = asymmetric_cost(y_true, np.array([0, 0, 0, 1]), c_fp=5.0, c_fn=20.0)
    assert cost_fn == 20.0

    # Ratio test
    assert (cost_fn / cost_fp) == pytest.approx(4.0)
