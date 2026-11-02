"""Calibration helpers for model outputs and probability interpretation."""
from __future__ import annotations
import math
from typing import Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_calibrator(raw_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling to raw model margins using validation data only."""
    scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if scores.shape[0] != y.shape[0] or scores.shape[0] == 0:
        raise ValueError("raw_scores and labels must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("raw_scores must contain only finite values")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("labels must contain both classes encoded as 0 and 1")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(scores, y)
    return calibrator


def bootstrap_brier_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
    split_strategy: str = "random",
) -> dict:
    """Return a percentile 95% bootstrap interval for held-out Brier score."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true and probabilities must have the same non-zero length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("labels must be binary and probabilities finite in [0, 1]")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")

    rng = np.random.default_rng(random_state)
    estimates = np.empty(n_bootstrap, dtype=float)
    n = y.size
    for i in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        estimates[i] = float(np.mean((p[sample] - y[sample]) ** 2))
    return {
        "method": "Nonparametric row bootstrap; percentile interval",
        "confidence_level": 0.95,
        "replicates": int(n_bootstrap),
        "estimate": float(np.mean((p - y) ** 2)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "note": (
            "Sampling uncertainty on this chronological holdout within the same 48-hour dataset; "
            "not an independent future-period guarantee."
            if split_strategy == "temporal"
            else "Sampling uncertainty on this held-out random split; not a future-period guarantee."
        ),
    }


def _calibration_group_summary(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "transactions": int(y.size),
        "fraud_transactions": int(y.sum()),
        "mean_predicted_probability": float(p.mean()) if p.size else None,
        "observed_fraud_rate": float(y.mean()) if y.size else None,
        "brier_score": float(np.mean((p - y) ** 2)) if y.size else None,
    }


def grouped_calibration_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amount: np.ndarray,
    time_seconds: np.ndarray,
    *,
    split_strategy: str = "random",
) -> dict:
    """Summarize held-out calibration by amount tertiles and dataset-time halves."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    amount = np.asarray(amount, dtype=float).reshape(-1)
    time_seconds = np.asarray(time_seconds, dtype=float).reshape(-1)
    if not (y.size == p.size == amount.size == time_seconds.size) or y.size == 0:
        raise ValueError("labels, probabilities, amount, and time must have equal non-zero lengths")
    if not all(np.isfinite(values).all() for values in (p, amount, time_seconds)):
        raise ValueError("probabilities and grouping values must be finite")

    report = {}
    time_group_name = "test_window_time_halves" if split_strategy == "temporal" else "dataset_time_halves"
    for name, values, labels, quantiles in (
        ("amount_tertiles", amount, ("low", "middle", "high"), (1 / 3, 2 / 3)),
        (time_group_name, time_seconds, ("earlier", "later"), (0.5,)),
    ):
        boundaries = np.unique(np.quantile(values, quantiles))
        codes = np.digitize(values, boundaries, right=True)
        report[name] = {}
        for code, label in enumerate(labels):
            mask = codes == code
            if mask.any():
                report[name][label] = _calibration_group_summary(y[mask], p[mask])
    grouping_note = (
        "Slices within the chronological held-out test window. These are descriptive groups from the same 48-hour source dataset."
        if split_strategy == "temporal"
        else "Descriptive slices from the same held-out random split. Time halves are not an out-of-time test."
    )
    return {"grouping_note": grouping_note, **report}


def model_output_to_score(raw_value: float, output_type: str = "raw_margin",
                          calibrator: Optional[Any] = None) -> float:
    """Convert one model output to a score in [0, 1], applying calibration if present."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"model output must be a finite number, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"model output must be finite, got {raw_value!r}")
    if calibrator is not None:
        score = float(calibrator.predict_proba(np.asarray([[value]], dtype=float))[0, 1])
    elif output_type == "probability":
        score = value
    elif output_type == "raw_margin":
        score = 1.0 / (1.0 + math.exp(-float(np.clip(value, -40.0, 40.0))))
    else:
        raise ValueError(f"Unsupported model output_type: {output_type!r}")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"converted model score must be finite and in [0, 1], got {score!r}")
    return score
