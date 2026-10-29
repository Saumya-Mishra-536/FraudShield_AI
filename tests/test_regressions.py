"""Additional regression tests for dataset loading and asymmetric objective."""
import numpy as np
import pandas as pd
import pytest

from data.load_kaggle_data import load_and_validate
from src.loss_functions import xgboost_asymmetric_objective


def make_dataset(rows=12):
    data = {"Time": np.arange(rows, dtype=float), "Amount": np.arange(rows, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = np.zeros(rows, dtype=float)
    data["Class"] = np.array([0, 1] * (rows // 2), dtype=int)
    return pd.DataFrame(data)


def test_loader_accepts_valid_dataset(tmp_path):
    path = tmp_path / "sample.csv"
    make_dataset().to_csv(path, index=False)
    loaded = load_and_validate(str(path))
    assert len(loaded) == 12
    assert loaded["Class"].sum() == 6


def test_loader_rejects_missing_feature(tmp_path):
    path = tmp_path / "bad.csv"
    df = make_dataset().drop(columns=["V7"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_and_validate(str(path))


def test_loader_rejects_missing_values(tmp_path):
    path = tmp_path / "null.csv"
    df = make_dataset()
    df.loc[0, "V3"] = np.nan
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_and_validate(str(path))


def test_asymmetric_objective_returns_finite_gradient_and_hessian():
    xgboost = pytest.importorskip("xgboost")
    objective = xgboost_asymmetric_objective(c_fp=5.0, c_fn=20.0)
    labels = np.array([0, 1, 0, 1], dtype=float)
    raw_scores = np.array([-2.0, -0.5, 1.0, 2.5], dtype=float)
    dmatrix = xgboost.DMatrix(np.zeros((4, 1)), label=labels)
    grad, hess = objective(raw_scores, dmatrix)
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()
    assert (hess > 0).all()
    assert grad.shape == labels.shape
    assert hess.shape == labels.shape
