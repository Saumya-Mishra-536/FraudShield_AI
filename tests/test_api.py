"""
API Integration & Explainability Endpoint Tests
================================================
Tests FastAPI routes with TestClient, verifying:
1. Health and model info endpoints.
2. Predict endpoint logic with unified decision engine.
3. Risk score endpoint containing both local SHAP explanations and global importance reference.
4. Dedicated /explain endpoint returning additive local SHAP values.
5. Global importance endpoint distinguishing tree-split gains from transaction-level SHAP.
6. Validation errors for missing, negative, and non-finite inputs (422 responses).
7. Validation errors for invalid query parameters (top_k, importance_type).
8. Decision consistency: /predict and /risk-score always agree.
9. Absence of hardcoded currency symbols in explanations.
"""
import math
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
        "output_type": "probability",
        "probability_calibrated": False,
        "threshold": 0.160,
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
    assert data["explainer_loaded"] is True
    assert data["calibrated_threshold"] is not None


def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "FraudShield" in data["model_name"]
    assert "cost_assumptions" in data
    assert data["cost_assumptions"]["cost_false_positive"] == 5.0
    assert data["cost_assumptions"]["cost_false_negative"] == 20.0
    assert "decision_thresholds" in data
    assert "fraud_decision_threshold" in data["decision_thresholds"]


def test_predict_endpoint(client):
    payload = sample_transaction_payload()
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert data["decision"] in ["ALLOW", "CHALLENGE", "BLOCK", "HARD_BLOCK"]
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["probability_calibrated"], bool)
    assert data["model_risk_score"] == data["fraud_probability"]
    assert data["score_semantics"] in {"calibrated_probability", "uncalibrated_model_risk_score"}


def test_predict_and_risk_score_never_contradict(client):
    """Verify predict and risk-score produce identical probability, is_fraud, and decision."""
    for test_args in [
        {"V14": 0.5, "Amount": 20.0},                  # Likely low risk
        {"V14": -3.5, "Amount": 150.0},                # Likely borderline
        {"V14": -8.0, "V4": 6.0, "Amount": 2500.0},    # High fraud risk
    ]:
        payload = sample_transaction_payload(**test_args)
        res_pred = client.post("/predict", json=payload).json()
        res_risk = client.post("/risk-score", json=payload).json()

        assert res_pred["fraud_probability"] == pytest.approx(res_risk["fraud_probability"], rel=1e-5)
        assert res_pred["is_fraud"] == res_risk["is_fraud"]
        assert res_pred["decision"] == res_risk["decision"]
        assert res_pred["risk_tier"] == res_risk["risk_tier"]


def test_risk_score_distinguishes_local_and_global_explanations(client):
    payload = sample_transaction_payload(V14=-7.0, V4=6.0, Amount=500.0)
    res = client.post("/risk-score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "risk_tier" in data
    assert "local_shap_drivers" in data
    assert len(data["local_shap_drivers"]) > 0
    assert "raw_margin_attribution" in data
    assert "base_value_log_odds" in data["raw_margin_attribution"]
    assert "threshold_used" in data
    if data["probability_calibrated"]:
        assert data["expected_exposure_amount_units"] == pytest.approx(data["fraud_probability"] * payload["Amount"], abs=0.01)
    else:
        assert data["expected_exposure_amount_units"] is None

    top_local = data["local_shap_drivers"][0]
    assert "shap_value" in top_local
    assert "effect" in top_local

    assert "global_importance_reference" in data
    assert len(data["global_importance_reference"]) > 0
    top_global = data["global_importance_reference"][0]
    assert "importance_score" in top_global

    # Ensure no hardcoded currency symbols
    explanation_text = " ".join(data["explanation"])
    assert "€" not in explanation_text
    assert "$" not in explanation_text


def test_explain_endpoint_local_attribution(client):
    payload = sample_transaction_payload()
    res = client.post("/explain?top_k=7", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["explanation_type"] == "local_shap_attribution"
    assert "base_value_log_odds" in data
    assert "top_risk_increasing_factors" in data
    assert "top_risk_decreasing_factors" in data
    assert "top_drivers_by_magnitude" in data
    assert len(data["top_drivers_by_magnitude"]) <= 7


def test_global_importance_endpoint(client):
    res = client.get("/features/global-importance?top_k=10&importance_type=gain")
    assert res.status_code == 200
    data = res.json()
    assert "global_gain_importance" in data["explanation_type"]
    assert len(data["top_features"]) == 10


def test_validation_rejects_negative_amount(client):
    payload = sample_transaction_payload(Amount=-15.0)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_validation_rejects_negative_time(client):
    payload = sample_transaction_payload(Time=-100.0)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_validation_rejects_missing_field(client):
    payload = sample_transaction_payload()
    del payload["V14"]
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_validation_rejects_invalid_importance_type(client):
    res = client.get("/features/global-importance?importance_type=invalid_metric")
    assert res.status_code == 422


def test_validation_rejects_out_of_range_top_k(client):
    res = client.get("/features/global-importance?top_k=999")
    assert res.status_code == 422
    res2 = client.get("/features/global-importance?top_k=0")
    assert res2.status_code == 422


def test_api_key_authentication_and_health_probe_exemption(client, monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_API_KEY", "ci-only-test-key")
    assert client.get("/model-info").status_code == 401
    assert client.get("/model-info", headers={"X-API-Key": "wrong"}).status_code == 401
    authorized = client.get("/model-info", headers={"X-API-Key": "ci-only-test-key"})
    assert authorized.status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_request_validation_response_does_not_echo_transaction_values(client):
    payload = sample_transaction_payload(V1="PRIVATE-TRANSACTION-VALUE")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "PRIVATE-TRANSACTION-VALUE" not in response.text
    assert "input" not in response.json()


def test_process_local_rate_limit_returns_retry_after(monkeypatch):
    from fastapi import FastAPI
    from src.api import APIProtectionMiddleware

    monkeypatch.delenv("FRAUDSHIELD_API_KEY", raising=False)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    isolated_app = FastAPI()
    isolated_app.add_middleware(APIProtectionMiddleware)

    @isolated_app.get("/probe")
    def probe():
        return {"ok": True}

    with TestClient(isolated_app) as isolated_client:
        assert isolated_client.get("/probe").status_code == 200
        limited = isolated_client.get("/probe")
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_production_mode_fails_closed_without_api_key(client, monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_ENV", "production")
    monkeypatch.delenv("FRAUDSHIELD_API_KEY", raising=False)
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/health/ready").status_code == 503
    assert client.get("/model-info").status_code == 503
