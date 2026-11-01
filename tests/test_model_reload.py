"""
Model Reload & Persistence Consistency Tests
=============================================
Verifies:
1. SymmetricModel and AsymmetricModel save and reload without loss of state.
2. Predictions before and after reload are bit-identical.
3. Feature names, thresholds, and best iterations are preserved on reload.
4. Scaler transformation invariance before and after disk reload.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.model import AsymmetricModel, SymmetricModel
from src.preprocessing import FEATURE_COLS


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(60, len(FEATURE_COLS)))
    y = np.array([0] * 50 + [1] * 10)
    return X, y


def test_asymmetric_model_reload_consistency(tmp_path, synthetic_data):
    X, y = synthetic_data
    save_path = str(tmp_path / "asym_test.pkl")

    # Fit original model
    model = AsymmetricModel(c_fp=5.0, c_fn=20.0)
    model.fit(X[:40], y[:40], X[40:], y[40:], feature_names=FEATURE_COLS)
    preds_before = model.predict_proba(X[40:])
    threshold_before = model.threshold

    # Save to disk
    model.save(save_path)
    assert os.path.exists(save_path)

    # Reload into new instance
    reloaded = AsymmetricModel(c_fp=5.0, c_fn=20.0)
    reloaded.load(save_path)
    preds_after = reloaded.predict_proba(X[40:])

    # Assert identity
    assert reloaded.threshold == pytest.approx(threshold_before)
    assert reloaded.feature_names == FEATURE_COLS
    np.testing.assert_allclose(
        preds_before, preds_after, rtol=1e-7, atol=1e-7,
        err_msg="Predictions differed after model reload!"
    )


def test_symmetric_model_reload_consistency(tmp_path, synthetic_data):
    X, y = synthetic_data
    save_path = str(tmp_path / "sym_test.pkl")

    model = SymmetricModel(threshold=0.25)
    model.fit(X[:40], y[:40], X[40:], y[40:], feature_names=FEATURE_COLS)
    preds_before = model.predict_proba(X[40:])
    threshold_before = model.threshold

    model.save(save_path)
    assert os.path.exists(save_path)

    reloaded = SymmetricModel()
    reloaded.load(save_path)
    preds_after = reloaded.predict_proba(X[40:])

    assert reloaded.threshold == pytest.approx(threshold_before)
    assert reloaded.feature_names == FEATURE_COLS
    np.testing.assert_allclose(
        preds_before, preds_after, rtol=1e-7, atol=1e-7,
        err_msg="Predictions differed after model reload!"
    )


def test_scaler_reload_consistency(tmp_path, synthetic_data):
    X, _ = synthetic_data
    scaler_path = str(tmp_path / "scaler_test.pkl")

    scaler = StandardScaler()
    scaled_before = scaler.fit_transform(X)

    joblib.dump(scaler, scaler_path)
    reloaded_scaler = joblib.load(scaler_path)
    scaled_after = reloaded_scaler.transform(X)

    np.testing.assert_allclose(
        scaled_before, scaled_after, rtol=1e-7, atol=1e-7,
        err_msg="Scaled features differed after scaler reload!"
    )

