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
