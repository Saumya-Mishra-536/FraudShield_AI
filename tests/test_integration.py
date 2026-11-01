"""
End-to-End Pipeline & Multi-Transaction Integration Tests
==========================================================
Verifies:
1. End-to-end integration: Raw Transaction -> Preprocessing -> Inference -> Decision Engine -> SHAP Explanation.
2. Complete parity between API endpoints and DecisionEngine across multiple diverse transaction profiles.
3. Verification that explanations correspond to injected anomalies.
4. Dynamic test set simulation produces exact held-out metrics from model_comparison.json at calibrated threshold.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app, load_artifacts, transaction_to_array, get_fraud_proba
from src.decision_engine import DecisionEngine
from src.explain import TransactionExplainer, generate_shap_narrative
from src.preprocessing import FEATURE_COLS, engineer_features


@pytest.fixture(scope="module")
def api_client():
    load_artifacts()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def artifacts():
    models_dir = os.path.join(ROOT, "models")
    asym_path = os.path.join(models_dir, "asymmetric_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    bundle = joblib.load(asym_path)
    scaler = joblib.load(scaler_path)
    explainer = TransactionExplainer(bundle["model"], bundle.get("feature_names", FEATURE_COLS))
    engine = DecisionEngine(fraud_threshold=float(bundle["threshold"]))
    return bundle, scaler, explainer, engine


TRANSACTION_PROFILES = [
    {
        "name": "legitimate_everyday_purchase",
        "payload": {**{f"V{i}": 0.05 for i in range(1, 29)}, "Amount": 34.50, "Time": 45000.0, "V14": 0.20, "V4": -0.10},
        "expected_fraud": False,
        "expected_tier_in": ["LOW", "MEDIUM"],
    },
    {
        "name": "high_risk_blatant_fraud",
        "payload": {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 1800.0, "Time": 7200.0, "V14": -8.5, "V4": 6.5, "V12": -7.0, "V10": -5.5},
        "expected_fraud": True,
        "expected_tier_in": ["HIGH", "CRITICAL"],
    },
    {
        "name": "micro_amount_card_testing",
        "payload": {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 0.50, "Time": 3600.0, "V14": -6.0, "V4": 4.5},
        "expected_fraud": True,
        "expected_tier_in": ["HIGH", "CRITICAL"],
    },
    {
        "name": "ambiguous_borderline_case",
        "payload": {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 420.0, "Time": 81000.0, "V14": -2.2, "V4": 1.8},
        "expected_fraud": None,  # May be either side of boundary
        "expected_tier_in": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    },
]


@pytest.mark.parametrize("profile", TRANSACTION_PROFILES)
def test_full_workflow_end_to_end_parity(api_client, artifacts, profile):
    """Verify that API and local pipeline compute identical decisions for every profile."""
    bundle, scaler, explainer, engine = artifacts
    payload = profile["payload"]

    # 1. Pipeline via Direct Python Code
    df_raw = pd.DataFrame([payload])
    df_eng = engineer_features(df_raw)
    X_scaled = scaler.transform(df_eng[FEATURE_COLS].values)

    import xgboost as xgb
    dmat = xgb.DMatrix(X_scaled, feature_names=FEATURE_COLS)
    raw = bundle["model"].predict(dmat, iteration_range=(0, bundle["model"].best_iteration + 1))[0]
    direct_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40))))
    direct_decision = engine.evaluate(direct_prob)
    direct_shap = explainer.explain_instance(X_scaled, raw_feature_dict=payload, top_k=3)

    # 2. Pipeline via REST API
    res_pred = api_client.post("/predict", json=payload).json()
    res_risk = api_client.post("/risk-score", json=payload).json()

    # Assert API outputs match each other
    assert res_pred["fraud_probability"] == pytest.approx(res_risk["fraud_probability"], rel=1e-5)
    assert res_pred["is_fraud"] == res_risk["is_fraud"]
    assert res_pred["decision"] == res_risk["decision"]
    assert res_pred["risk_tier"] == res_risk["risk_tier"]

    # Assert API outputs match Direct Pipeline
    assert res_pred["fraud_probability"] == pytest.approx(direct_prob, abs=1e-5)
    assert res_pred["is_fraud"] == direct_decision.is_fraud
    assert res_pred["decision"] == direct_decision.decision
    assert res_pred["risk_tier"] == direct_decision.risk_tier

    # Check expectation bounds
    if profile["expected_fraud"] is not None:
        assert res_pred["is_fraud"] is profile["expected_fraud"], (
            f"Profile {profile['name']} expected fraud={profile['expected_fraud']} got {res_pred['is_fraud']}"
        )
    assert res_pred["risk_tier"] in profile["expected_tier_in"]

    # Verify SHAP attributions correspond to injected features
    if profile["name"] == "high_risk_blatant_fraud":
        risk_drivers = [f["feature"] for f in direct_shap["top_risk_increasing_factors"]]
        # Injected anomaly V14 must be among primary risk-increasing drivers
        assert "V14" in risk_drivers or "V4" in risk_drivers, (
            f"Expected V14 or V4 to be detected as risk drivers, got {risk_drivers}"
        )


def test_dynamic_test_predictions_match_saved_model_comparison():
    """Verify that outputs/test_predictions.npz exactly matches outputs/model_comparison.json."""
    metrics_path = os.path.join(ROOT, "outputs", "model_comparison.json")
    preds_path = os.path.join(ROOT, "outputs", "test_predictions.npz")

    missing = [path for path in (metrics_path, preds_path) if not os.path.isfile(path)]
    if missing:
        pytest.fail(
            "Required evaluation artifacts are missing: " + ", ".join(missing),
            pytrace=False,
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    preds_data = np.load(preds_path)
    y_test = preds_data["y_test"]
    asym_proba = preds_data["asym_proba"]

    a_m = metrics["asymmetric_model"]
    tau = a_m["threshold"]

    # Compute empirical test confusion matrix at saved threshold
    pred_binary = (asym_proba >= tau).astype(int)
    tp = int(np.sum((pred_binary == 1) & (y_test == 1)))
    fp = int(np.sum((pred_binary == 1) & (y_test == 0)))
    fn = int(np.sum((pred_binary == 0) & (y_test == 1)))
    tn = int(np.sum((pred_binary == 0) & (y_test == 0)))

    assert tp == a_m["true_positives"]
    assert fp == a_m["false_positives"]
    assert fn == a_m["false_negatives"]
    assert tn == a_m["true_negatives"]

    calc_cost = 5.0 * fp + 20.0 * fn
    assert calc_cost == pytest.approx(a_m["total_asymmetric_cost"])
