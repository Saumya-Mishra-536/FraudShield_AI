# API examples and research notes

## Start locally

Use Python 3.12 and the pinned packages in `requirements.txt`. The API loads the saved asymmetric model, scaler, and SHAP explainer at startup.

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Check service health and readiness:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/ready
```

`/health/ready` returns HTTP 503 when required model artifacts are unavailable. Health probes remain public. When `FRAUDSHIELD_API_KEY` is configured, other endpoints require `X-API-Key`; local development leaves authentication disabled if that variable is unset. Set `FRAUDSHIELD_ENV=production` to make a missing API key fail readiness and reject protected requests. The in-process limiter defaults to 60 requests per client per minute; multi-worker deployments need a shared gateway or limiter.

## Prediction request

The request requires V1–V28 plus Amount and Time. This Python example supplies all 30 fields:

```python
import os
import requests

transaction = {f"V{i}": 0.0 for i in range(1, 29)}
transaction.update({"Amount": 120.50, "Time": 36000.0, "V14": -4.5, "V4": 3.2})
headers = {}
if os.getenv("FRAUDSHIELD_API_KEY"):
    headers["X-API-Key"] = os.environ["FRAUDSHIELD_API_KEY"]
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=transaction,
    headers=headers,
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

Example response shape from the refreshed artifact (values are illustrative):

```json
{
  "fraud_probability": 0.03,
  "model_risk_score": 0.03,
  "probability_calibrated": true,
  "score_semantics": "calibrated_probability",
  "calibration_method": "Platt scaling fitted on full validation split",
  "is_fraud": false,
  "decision": "CHALLENGE",
  "risk_tier": "MEDIUM",
  "threshold_used": 0.0492,
  "model_version": "asymmetric-v1.1"
}
```

The `fraud_probability` field is retained for compatibility. Treat it as a posterior only when `probability_calibrated` is true. The `model_risk_score` field always identifies the numerical score used by the configured action threshold.

## Detailed risk response

Send the same JSON to `POST /risk-score`. It includes SHAP explanations and global feature references. `expected_exposure_amount_units` is calculated from calibrated probability times Amount for the refreshed artifacts; it remains an exposure proxy, not expected chargeback loss.

## Calibration and artifact refresh

The asymmetric model selects its operating threshold using five-fold out-of-fold Platt calibrated validation scores, then refits the served Platt calibrator on all validation rows. Held-out test data is used only for evaluation. Training writes a bootstrap 95% Brier interval and amount/time subgroup summaries to `outputs/calibration_evaluation.json`, alongside the reliability curve at `outputs/plots/calibration_curve.png`. The bootstrap and slices quantify uncertainty on the same random test split; they do not substitute for later-time or independent-dataset evaluation.

Run the **Refresh calibrated artifacts** workflow manually from GitHub Actions to regenerate and commit the primary model, scaler, evaluation, and calibration artifacts. CI also trains a separate chronological 70/10/20 evaluation and verifies that the final test window follows the training and validation windows in Time. Its report is stored under `outputs/time_holdout/`; it covers a later segment of the same 48-hour dataset, not a newer independent dataset.

## Research limitations

- The primary metrics use stratified random splitting. A separate chronological holdout trains on the earliest 70%, uses the next 10% for calibration and threshold selection, and evaluates the latest 20%; this is still only a segment of the same roughly 48-hour dataset, not an independent future period.
- The dataset is highly imbalanced and contains few fraud examples; precision, recall, and calibration estimates have substantial sampling uncertainty.
- The $5 false-positive and $20 false-negative costs are illustrative. They are not derived from observed issuer loss, recoveries, investigation, customer churn, or jurisdiction-specific fees.
- Probability times transaction Amount is only an exposure proxy. It does not estimate net chargeback loss or account for recoveries and operational costs.
- The source rows contain anonymized PCA features `V1`–`V28`. Ordinary bank transaction fields cannot be sent to this model without reproducing the original PCA transform and feature preparation, or training a replacement model on an institution-owned feature set. The transform parameters and source mapping are not included here.
- The PCA components do not identify protected groups; this project does not evaluate fairness across protected groups or performance across merchants, card products, regions, and customer segments.
- Platt calibration assumes the relationship between model margins and outcome remains stable. The dashboard can route live scoring through FastAPI by setting `FRAUDSHIELD_API_URL`; configure `FRAUDSHIELD_API_KEY` in both server environments when authentication is enabled. Production use would require ongoing calibration, drift, and subgroup monitoring.
- This is a research and portfolio demonstration, not an autonomous payment authorization system.
