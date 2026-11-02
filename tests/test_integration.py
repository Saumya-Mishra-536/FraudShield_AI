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
from src.calibration import model_output_to_score
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
    direct_prob = model_output_to_score(raw, output_type=bundle.get("output_type", "raw_margin"), calibrator=bundle.get("calibrator"))
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



def test_fresh_calibration_report_and_cross_fitted_threshold_artifact():
    """CI training must emit calibration uncertainty, subgroup slices, and a cross-fit method."""
    report_path = os.path.join(ROOT, "outputs", "calibration_evaluation.json")
    metrics_path = os.path.join(ROOT, "outputs", "model_comparison.json")
    bundle_path = os.path.join(ROOT, "models", "asymmetric_model.pkl")
    for path in (report_path, metrics_path, bundle_path):
        if not os.path.isfile(path):
            pytest.fail(f"Required freshly generated artifact is missing: {path}", pytrace=False)

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    interval = report["brier_score_bootstrap"]
    assert interval["replicates"] >= 100
    assert interval["lower_95"] <= interval["estimate"] <= interval["upper_95"]
    groups = report["held_out_groups"]
    assert set(groups["amount_tertiles"]) == {"low", "middle", "high"}
    assert set(groups["dataset_time_halves"]) == {"earlier", "later"}
    assert "not an out-of-time test" in groups["grouping_note"]

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    assert "calibration_evaluation" in metrics
    assert metrics["metadata"]["threshold_selection_method"].startswith("5-fold cross-fitted")

    bundle = joblib.load(bundle_path)
    assert bundle["calibrator"] is not None
    assert "cross-fitted" in bundle["threshold_selection_method"]



def test_temporal_holdout_metrics_use_disjoint_later_windows():
    """A separate report evaluates the last chronological segment of the source dataset."""
    metrics_path = os.path.join(ROOT, "outputs", "time_holdout", "model_comparison.json")
    calibration_path = os.path.join(ROOT, "outputs", "time_holdout", "calibration_evaluation.json")
    for path in (metrics_path, calibration_path):
        if not os.path.isfile(path):
            pytest.fail(f"Required forward-time artifact is missing: {path}", pytrace=False)

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    metadata = metrics["metadata"]
    assert metadata["split_strategy"] == "temporal"
    ranges = metadata["split_time_ranges"]
    assert ranges["train"]["end_seconds"] < ranges["validation"]["start_seconds"]
    assert ranges["validation"]["end_seconds"] < ranges["test"]["start_seconds"]
    counts = metadata["split_class_counts"]
    assert counts["train"]["1"] > 0
    assert counts["validation"]["1"] > 0
    assert counts["test"]["1"] > 0

    with open(calibration_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    interval = report["brier_score_bootstrap"]
    assert interval["lower_95"] <= interval["estimate"] <= interval["upper_95"]
    assert "chronological holdout" in interval["note"]
    assert "same 48-hour source dataset" in report["held_out_groups"]["grouping_note"]
    assert "test_window_time_halves" in report["held_out_groups"]

def test_dashboard_live_scoring_client_calls_real_risk_score_endpoint(api_client, monkeypatch):
    """The dashboard HTTP client sends its payload and key through the actual API route."""
    import httpx

    from src.dashboard_api import request_risk_score

    monkeypatch.setenv("FRAUDSHIELD_API_KEY", "integration-test-key")
    transaction = TRANSACTION_PROFILES[0]["payload"]

    def forward_to_fastapi(request):
        assert request.url.path == "/risk-score"
        assert request.headers.get("X-API-Key") == "integration-test-key"
        response = api_client.post(
            request.url.path,
            json=json.loads(request.content),
            headers={"X-API-Key": request.headers["X-API-Key"]},
        )
        return httpx.Response(response.status_code, json=response.json())

    result, latency_ms = request_risk_score(
        "http://fraudshield.test",
        "integration-test-key",
        transaction,
        transport=httpx.MockTransport(forward_to_fastapi),
    )

    assert result["model_version"] == "asymmetric-v1.2"
    assert result["decision"] in {"ALLOW", "CHALLENGE", "BLOCK", "HARD_BLOCK"}
    assert result["probability_calibrated"] is True
    assert result["top_risk_increasing_factors"] or result["top_risk_decreasing_factors"]
    assert latency_ms >= 0

