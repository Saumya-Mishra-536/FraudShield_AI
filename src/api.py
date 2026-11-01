"""
Fraud Detection REST API — Kaggle CC Fraud Dataset Schema
==========================================================
FastAPI service exposing:
  - POST /predict      → Unified fraud decision, tier, and probability
  - POST /risk-score   → Detailed risk classification + pure SHAP explanation
  - POST /explain      → Direct instance-level SHAP decomposition
  - GET  /features/global-importance → Global XGBoost gain rankings
  - GET  /health       → Service health check
  - GET  /model-info   → Model metadata, decision thresholds, and cost configuration

Features:
  - Unified decision logic: `/predict` and `/risk-score` never contradict.
  - 100% SHAP-driven explanations: heuristic rules and currency assumptions removed.
  - Robust Pydantic v2 validation with finite float checks and boundary guards.
  - Configurable model path via MODEL_DIR environment variable.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Ensure silent Matplotlib config directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.calibration import model_output_to_score
from src.decision_engine import DecisionEngine, RiskDecision
from src.explain import TransactionExplainer, generate_shap_narrative
from src.loss_functions import DEFAULT_C_FN, DEFAULT_C_FP
from src.preprocessing import FEATURE_COLS, engineer_features

# Environment-configurable paths
MODELS_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
ASYM_MODEL_PATH = os.path.join(MODELS_DIR, "asymmetric_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Global runtime state
_model: Optional[Dict[str, Any]] = None
_scaler: Any = None
_feature_names: Optional[List[str]] = None
_explainer: Optional[TransactionExplainer] = None
_decision_engine: Optional[DecisionEngine] = None
_startup_error: Optional[str] = None


# ─── Pydantic Request & Response Schemas ─────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Kaggle Credit Card Fraud transaction payload.
    Features:
      V1-V28: PCA anonymized components (dimensionless, zero-centered).
      Amount: Transaction amount in dataset units (non-negative).
      Time: Elapsed seconds since dataset epoch (non-negative).
    """
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    Amount: float = Field(
        ...,
        ge=0.0,
        le=10_000_000.0,
        description="Transaction amount in dataset units (currency is unspecified).",
    )
    Time: float = Field(
        ...,
        ge=0.0,
        le=100_000_000.0,
        description="Seconds elapsed since dataset start.",
    )

    @model_validator(mode="after")
    def validate_finite_numbers(self) -> TransactionRequest:
        """Reject non-finite inputs (NaN, Infinity, -Infinity)."""
        for field_name in [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]:
            val = getattr(self, field_name)
            if val is None or not math.isfinite(val):
                raise ValueError(f"Field '{field_name}' must be a finite real number, got {val}")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
                "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
                "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
                "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
                "Amount": 149.62,
                "Time": 406.0,
            }
        }
    }


class PredictResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model score in [0, 1]; treat as posterior only when probability_calibrated is true.")
    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold_used.")
    decision: str = Field(..., description="Action: ALLOW, CHALLENGE, BLOCK, or HARD_BLOCK.")
    risk_tier: str = Field(..., description="Risk tier: LOW, MEDIUM, HIGH, or CRITICAL.")
    threshold_used: float = Field(..., description="Primary calibrated decision threshold.")
    probability_calibrated: bool
    score_semantics: str
    calibration_method: Optional[str] = None
    model_version: str = Field(..., description="Active model deployment version.")


class RiskScoreResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model score in [0, 1]; consult probability_calibrated before treating it as a posterior.")
    probability_calibrated: bool
    score_semantics: str
    calibration_method: Optional[str] = None
    is_fraud: bool
    risk_tier: str
    risk_score: float = Field(..., description="Model score scaled to 0-100.")
    decision: str
    explanation: List[str] = Field(default_factory=list, description="SHAP-derived narrative explanation.")
    top_risk_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Top features by global gain.")
    local_shap_drivers: List[Dict[str, Any]] = Field(default_factory=list, description="Instance-specific local SHAP attributions.")
    global_importance_reference: List[Dict[str, Any]] = Field(default_factory=list, description="Dataset-level global split importance reference.")
    expected_exposure_amount_units: Optional[float] = Field(None, description="P * Amount, returned only when a saved calibrator is available.")
    recommendation: str = Field(..., description="Operational action recommendation.")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    explainer_loaded: bool
    calibrated_threshold: Optional[float] = None
    probability_calibrated: bool = False
    calibration_method: Optional[str] = None
    startup_error: Optional[str] = None


# ─── Application Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudShield AI — Cost-Sensitive Fraud Detection API",
    description=(
        "Production-ready inference and explainability microservice for credit card fraud detection. "
        "Employs an asymmetric XGBoost model with Kahneman-Tversky loss weighting and SHAP TreeExplainer attributions."
    ),
    version="1.1.0",
)


def load_artifacts() -> None:
    """Load model, scaler, decision engine, and explainer from disk."""
    global _model, _scaler, _feature_names, _explainer, _decision_engine

    if not os.path.exists(ASYM_MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at '{ASYM_MODEL_PATH}'. Run: python -m src.train"
        )
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"Scaler artifact not found at '{SCALER_PATH}'. Run: python -m src.train"
        )

    bundle = joblib.load(ASYM_MODEL_PATH)
    _model = bundle
    _scaler = joblib.load(SCALER_PATH)
    _feature_names = bundle.get("feature_names", FEATURE_COLS)

    # Initialize unified decision engine with the calibrated threshold
    calibrated_t = float(bundle.get("threshold", 0.160))
    _decision_engine = DecisionEngine(fraud_threshold=calibrated_t)

    # Initialize SHAP TreeExplainer on booster
    _explainer = TransactionExplainer(_model["model"], _feature_names)
    import logging
    logging.getLogger("fraudshield.api").info(
        "model_artifacts_loaded", extra={"threshold": calibrated_t,
        "probability_calibrated": bool(bundle.get("probability_calibrated", bundle.get("calibrator") is not None)),
        "calibration_method": bundle.get("calibration_method")}
    )


@app.on_event("startup")
async def startup_event() -> None:
    global _startup_error
    import logging
    try:
        load_artifacts()
        _startup_error = None
    except (RuntimeError, OSError, ValueError) as exc:
        _startup_error = str(exc)
        logging.getLogger("fraudshield.api").exception(
            "model_artifact_load_failed", extra={"model_dir": MODELS_DIR}
        )


# ─── Utilities ────────────────────────────────────────────────────────────────

def transaction_to_array(tx: TransactionRequest) -> np.ndarray:
    """Convert request payload to engineered and scaled feature matrix."""
    if _scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model scaler is not initialized. Ensure models are trained and present in models/.",
        )

    row = {f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)}
    row["Amount"] = float(tx.Amount)
    row["Time"] = float(tx.Time)

    df = pd.DataFrame([row])
    df = engineer_features(df)
    X = df[FEATURE_COLS].values
    return _scaler.transform(X)


def get_fraud_proba(X_scaled: np.ndarray) -> float:
    """Predict fraud probability using the model's best boosting iteration."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train model via: python -m src.train",
        )

    import xgboost as xgb
    dmat = xgb.DMatrix(X_scaled, feature_names=_feature_names)
    booster = _model["model"]
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        raw = booster.predict(dmat, iteration_range=(0, best_iteration + 1))
    else:
        raw = booster.predict(dmat)

    return model_output_to_score(raw[0], output_type=_model.get("output_type", "raw_margin"), calibrator=_model.get("calibrator"))


def get_top_risk_factors(X_scaled: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top global features by XGBoost gain importance."""
    if _model is None:
        return []
    scores = _model["model"].get_score(importance_type="gain")
    feature_names = _feature_names or FEATURE_COLS
    factors = [
        {
            "feature": f,
            "scaled_value": round(float(X_scaled[0, i]), 4),
            "importance_gain": round(scores.get(f, 0.0), 4),
        }
        for i, f in enumerate(feature_names)
    ]
    factors.sort(key=lambda x: x["importance_gain"], reverse=True)
    return factors[:top_k]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Service health and artifact availability check."""
    loaded = _model is not None and _scaler is not None and _explainer is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "explainer_loaded": _explainer is not None,
        "calibrated_threshold": _decision_engine.threshold if _decision_engine else None,
        "probability_calibrated": bool(_model and _model.get("probability_calibrated", _model.get("calibrator") is not None)),
        "calibration_method": _model.get("calibration_method") if _model else None,
        "startup_error": _startup_error,
    }


@app.get("/health/ready", tags=["System"])
def readiness():
    """Readiness probe for deployment platforms; 503 until inference artifacts load."""
    if _model is None or _scaler is None or _explainer is None or _decision_engine is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Model artifacts are not loaded.")
    return {"status": "ready"}


@app.get("/model-info", tags=["System"])
def model_info():
    """Model operational metadata, cost assumptions, and decision boundaries."""
    if _model is None or _decision_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )
    return {
        "model_name": "FraudShield Asymmetric Cost-Sensitive XGBoost",
        "probability_calibrated": bool(_model.get("probability_calibrated", _model.get("calibrator") is not None)),
        "score_semantics": "calibrated_probability" if _model.get("probability_calibrated", _model.get("calibrator") is not None) else "uncalibrated_model_risk_score",
        "calibration_method": _model.get("calibration_method"),
        "version": "1.1.0",
        "dataset": "Kaggle Credit Card Fraud Detection (MLG-ULB, 284,807 transactions)",
        "decision_thresholds": {
            "fraud_decision_threshold": _decision_engine.threshold,
            "challenge_threshold": _decision_engine.threshold_challenge,
            "hard_block_threshold": _decision_engine.threshold_hard_block,
            "theoretical_bayes_threshold": round(
                DEFAULT_C_FP / (DEFAULT_C_FP + DEFAULT_C_FN), 4
            ),
        },
        "cost_assumptions": {
            "cost_false_positive": DEFAULT_C_FP,
            "cost_false_negative": DEFAULT_C_FN,
            "asymmetry_ratio": round(DEFAULT_C_FN / DEFAULT_C_FP, 2),
            "note": "Illustrative units representing customer friction vs direct financial loss.",
        },
        "feature_count": len(_feature_names or FEATURE_COLS),
        "features": _feature_names or FEATURE_COLS,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(tx: TransactionRequest):
    """
    Unified fraud prediction endpoint.
    Guarantees strict alignment: decision and is_fraud are determined by the unified DecisionEngine.
    """
    if _decision_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Decision engine not initialized.")

    X = transaction_to_array(tx)
    proba = get_fraud_proba(X)
    decision = _decision_engine.evaluate(proba)

    probability_calibrated = bool(_model and _model.get("probability_calibrated", _model.get("calibrator") is not None))
    return {
        "fraud_probability": round(decision.fraud_probability, 6),
        "model_risk_score": round(decision.fraud_probability, 6),
        "probability_calibrated": probability_calibrated,
        "score_semantics": "calibrated_probability" if probability_calibrated else "uncalibrated_model_risk_score",
        "calibration_method": _model.get("calibration_method") if _model else None,
        "is_fraud": decision.is_fraud,
        "decision": decision.decision,
        "risk_tier": decision.risk_tier,
        "threshold_used": decision.threshold_used,
        "model_version": "asymmetric-v1.1",
    }


@app.post("/risk-score", response_model=RiskScoreResponse, tags=["Prediction"])
def risk_score(tx: TransactionRequest):
    """
    Detailed operational risk assessment with SHAP instance-level attribution.
    Contains no heuristic rules or currency assumptions.
    """
    if _decision_engine is None or _explainer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model/Explainer not initialized.")

    X = transaction_to_array(tx)
    proba = get_fraud_proba(X)
    decision = _decision_engine.evaluate(proba)

    # Local SHAP attribution
    raw_dict = {f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)}
    raw_dict["Amount"] = tx.Amount
    raw_dict["Time"] = tx.Time
    shap_res = _explainer.explain_instance(X, raw_feature_dict=raw_dict, top_k=5)

    # Generate SHAP-based narrative
    explanation = generate_shap_narrative(shap_res, decision.decision, top_k=3)
    top_factors = get_top_risk_factors(X, top_k=5)
    global_ref = _explainer.get_global_feature_importance(importance_type="gain", top_k=5)

    probability_calibrated = bool(_model and _model.get("probability_calibrated", _model.get("calibrator") is not None))
    expected_exposure = round(proba * tx.Amount, 2) if probability_calibrated else None

    return {
        "fraud_probability": round(decision.fraud_probability, 6),
        "model_risk_score": round(decision.fraud_probability, 6),
        "probability_calibrated": probability_calibrated,
        "score_semantics": "calibrated_probability" if probability_calibrated else "uncalibrated_model_risk_score",
        "calibration_method": _model.get("calibration_method") if _model else None,
        "is_fraud": decision.is_fraud,
        "risk_tier": decision.risk_tier,
        "risk_score": round(decision.fraud_probability * 100.0, 1),
        "decision": decision.decision,
        "explanation": explanation,
        "top_risk_factors": top_factors,
        "local_shap_drivers": shap_res["top_drivers_by_magnitude"],
        "global_importance_reference": global_ref["top_features"],
        "expected_exposure_amount_units": expected_exposure,
        "recommendation": decision.recommendation,
        "model_version": "asymmetric-v1.1",
    }


@app.post("/explain", tags=["Explainability"])
def explain_transaction(
    tx: TransactionRequest,
    top_k: int = Query(default=10, ge=1, le=31, description="Number of top driving features to highlight."),
):
    """
    Detailed instance-level local SHAP explanation.
    Decomposes log-odds: logit(p) = base_value + sum(shap_values).
    """
    if _explainer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="SHAP Explainer not initialized.")

    X = transaction_to_array(tx)
    raw_dict = {f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)}
    raw_dict["Amount"] = tx.Amount
    raw_dict["Time"] = tx.Time
    return _explainer.explain_instance(X, raw_feature_dict=raw_dict, top_k=top_k)


@app.get("/features/global-importance", tags=["Explainability"])
def global_feature_importance(
    importance_type: str = Query(
        default="gain",
        pattern="^(gain|weight|cover|total_gain|total_cover)$",
        description="XGBoost split evaluation metric.",
    ),
    top_k: int = Query(default=15, ge=1, le=31, description="Number of top features to return."),
):
    """
    Model-wide global feature importance across all boosted trees.
    Distinguished from local instance-level SHAP attributions.
    """
    if _explainer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Explainer not initialized.")
    return _explainer.get_global_feature_importance(importance_type=importance_type, top_k=top_k)


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
