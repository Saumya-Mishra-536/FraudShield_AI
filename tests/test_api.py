"""
API Integration & Explainability Endpoint Tests
================================================
Tests FastAPI routes with TestClient, verifying:
1. Health and model info endpoints.
2. Predict endpoint logic.
3. Risk score endpoint containing both local SHAP explanations and global importance reference.
4. Dedicated /explain endpoint returning additive local SHAP values.
5. Global importance endpoint distinguishing tree-split gains from transaction-level SHAP.
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
import xgboost as xgb
from fastapi.testclient import TestClient

from src.api import app, load_artifacts, ASYM_MODEL_PATH, SCALER_PATH, MODELS_DIR
from src.preprocessing import FEATURE_COLS


def ensure_test_artifacts():
    """Ensure artifacts exist so API tests pass in artifact-free CI environments."""
    if os.path.exists(ASYM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler = StandardScaler()
    X_dummy = np.random.randn(20, len(FEATURE_COLS))
    scaler.fit(X_dummy)
    joblib.dump(scaler, SCALER_PATH)

    y_dummy = np.array([0] * 16 + [1] * 4)
    dtrain = xgb.DMatrix(X_dummy, label=y_dummy, feature_names=FEATURE_COLS)
    params = {"max_depth": 3, "eta": 0.1, "objective": "binary:logistic"}
    booster = xgb.train(params, dtrain, num_boost_round=5)
    bundle = {
        "model": booster,
        "threshold": 0.20,
        "feature_names": FEATURE_COLS,
    }
    joblib.dump(bundle, ASYM_MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    ensure_test_artifacts()
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def sample_transaction_payload(**overrides):
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload.update({
        "Amount": 120.50,
        "Time": 36000.0,
        "V14": -4.5,
        "V4": 3.2,
    })
    payload.update(overrides)
    return payload


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True


def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "Asymmetric XGBoost"
    assert "cost_config" in data
    assert data["cost_config"]["C_FP_assumption_units"] == 5.0
    assert data["cost_config"]["C_FN_assumption_units"] == 20.0


def test_predict_endpoint(client):
    payload = sample_transaction_payload()
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert data["decision"] in ["ALLOW", "BLOCK"]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_risk_score_distinguishes_local_and_global_explanations(client):
    payload = sample_transaction_payload(V14=-7.0, V4=6.0, Amount=500.0)
    res = client.post("/risk-score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "risk_tier" in data
    assert "local_shap_drivers" in data
    assert len(data["local_shap_drivers"]) > 0
    # Verify local SHAP drivers have shap_value and effect
    top_local = data["local_shap_drivers"][0]
    assert "shap_value" in top_local
    assert "effect" in top_local

    assert "global_importance_reference" in data
    assert len(data["global_importance_reference"]) > 0
    top_global = data["global_importance_reference"][0]
    assert "importance_score" in top_global


def test_explain_endpoint_local_attribution(client):
    payload = sample_transaction_payload()
    res = client.post("/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["explanation_type"] == "local_shap_attribution"
    assert "base_value_log_odds" in data
    assert "top_risk_increasing_factors" in data
    assert "top_risk_decreasing_factors" in data
    assert "top_drivers_by_magnitude" in data


def test_global_importance_endpoint(client):
    res = client.get("/features/global-importance?top_k=10")
    assert res.status_code == 200
    data = res.json()
    assert "global_gain_importance" in data["explanation_type"]
    assert len(data["top_features"]) == 10
