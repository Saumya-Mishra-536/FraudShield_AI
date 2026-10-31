"""
Data-Free Smoke & Reproducibility Tests
=======================================
Runs fast in lightweight CI environments without requiring the full 150MB Kaggle dataset.
Tests:
1. End-to-end preprocessing on synthetic transaction data.
2. Training Symmetric and Asymmetric models on synthetic data.
3. Decision threshold tuning on synthetic validation splits.
4. Explainer generation and local attribution.
5. Deterministic reproducibility: same random seed yields identical predictions.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pytest

from src.explain import TransactionExplainer
from src.loss_functions import DEFAULT_C_FN, DEFAULT_C_FP
from src.model import AsymmetricModel, SymmetricModel
from src.preprocessing import FEATURE_COLS, preprocess


def generate_synthetic_transactions(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate minimal valid transactions matching Kaggle creditcard schema."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, size=n_samples)
    data["Amount"] = rng.exponential(scale=50.0, size=n_samples)
    data["Time"] = rng.uniform(0, 172800, size=n_samples)
    # Ensure at least 10 fraud samples for stratified splitting & SMOTE
    labels = np.zeros(n_samples, dtype=int)
    labels[:15] = 1
    rng.shuffle(labels)
    data["Class"] = labels
    return pd.DataFrame(data)


def test_synthetic_end_to_end_pipeline(tmp_path):
    """Verify complete train-tune-evaluate flow runs end-to-end on synthetic data."""
    df = generate_synthetic_transactions(n_samples=150, seed=123)
    scaler_file = str(tmp_path / "scaler.pkl")

    prepared = preprocess(
        df,
        test_size=0.2,
        val_size=0.2,
        apply_smote=False,
        random_state=42,
        scaler_path=scaler_file,
    )

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_val, y_val = prepared["X_val"], prepared["y_val"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    # 1. Train Symmetric
    sym = SymmetricModel(threshold=0.5)
    sym.fit(X_train, y_train, X_val, y_val, prepared["feature_names"])
    sym_t = sym.tune_threshold(X_val, y_val, DEFAULT_C_FP, DEFAULT_C_FN)
    assert 0.0 < sym_t < 1.0
    sym_preds = sym.predict(X_test)
    assert sym_preds.shape == (len(X_test),)

    # 2. Train Asymmetric
    asym = AsymmetricModel(c_fp=5.0, c_fn=20.0)
    asym.fit(X_train, y_train, X_val, y_val, prepared["feature_names"])
    assert 0.0 < asym.threshold < 1.0
    asym_preds = asym.predict(X_test)
    assert asym_preds.shape == (len(X_test),)

    # 3. Explainability check
    explainer = TransactionExplainer(asym.model, prepared["feature_names"])
    exp_res = explainer.explain_instance(X_test[:1])
    assert "top_risk_increasing_factors" in exp_res
    assert "base_value_log_odds" in exp_res


def test_deterministic_reproducibility():
    """Verify that two training runs with the same seed yield bit-identical predictions."""
    df = generate_synthetic_transactions(n_samples=100, seed=999)

    prep1 = preprocess(df, test_size=0.25, val_size=0.25, apply_smote=False, random_state=42)
    m1 = AsymmetricModel(c_fp=5.0, c_fn=20.0)
    m1.fit(prep1["X_train"], prep1["y_train"], prep1["X_val"], prep1["y_val"], prep1["feature_names"])
    p1 = m1.predict_proba(prep1["X_test"])

    prep2 = preprocess(df, test_size=0.25, val_size=0.25, apply_smote=False, random_state=42)
    m2 = AsymmetricModel(c_fp=5.0, c_fn=20.0)
    m2.fit(prep2["X_train"], prep2["y_train"], prep2["X_val"], prep2["y_val"], prep2["feature_names"])
    p2 = m2.predict_proba(prep2["X_test"])

    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5, err_msg="Model predictions are non-deterministic across identical seeds!")
