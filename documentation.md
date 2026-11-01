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

`/health/ready` returns HTTP 503 when required model artifacts are unavailable. OpenAPI docs are available at `/docs`.

## Prediction request

The request requires V1–V28 plus Amount and Time. This Python example supplies all 30 fields:

```python
import requests

transaction = {f"V{i}": 0.0 for i in range(1, 29)}
transaction.update({"Amount": 120.50, "Time": 36000.0, "V14": -4.5, "V4": 3.2})
response = requests.post("http://127.0.0.1:8000/predict", json=transaction, timeout=30)
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
  "calibration_method": "Platt scaling fitted on validation raw margins",
  "is_fraud": false,
  "decision": "CHALLENGE",
  "risk_tier": "MEDIUM",
  "threshold_used": 0.0483,
  "model_version": "asymmetric-v1.1"
}
```

The `fraud_probability` field is retained for compatibility. Treat it as a posterior only when `probability_calibrated` is true. The `model_risk_score` field always identifies the numerical score used by the configured action threshold.

## Detailed risk response

Send the same JSON to `POST /risk-score`. It includes SHAP explanations and global feature references. `expected_exposure_amount_units` is calculated from calibrated probability times Amount for the refreshed artifacts; it remains an exposure proxy, not expected chargeback loss.

## Calibration and artifact refresh

The asymmetric model trains a Platt scaling logistic regression on validation raw margins only. The operating threshold is selected on those calibrated validation scores. Held-out test data is used to report Brier score and produce `outputs/plots/calibration_curve.png`; it is not used to fit calibration.

Run the **Refresh calibrated artifacts** workflow manually from GitHub Actions to regenerate and commit model, scaler, evaluation, and calibration artifacts. The workflow verifies generated artifacts and runs integration tests before committing them. Until that workflow completes, older artifacts remain marked uncalibrated.

## Research limitations

- This repository uses stratified random splitting rather than forward-in-time evaluation. The dataset spans about 48 hours, so performance may be optimistic for future periods.
- The dataset is highly imbalanced and contains few fraud examples; precision, recall, and calibration estimates have substantial sampling uncertainty.
- The $5 false-positive and $20 false-negative costs are illustrative. They are not derived from observed issuer loss, recoveries, investigation, customer churn, or jurisdiction-specific fees.
- Probability times transaction Amount is only an exposure proxy. It does not estimate net chargeback loss or account for recoveries and operational costs.
- The PCA components are anonymized; this project does not evaluate fairness across protected groups or performance across merchants, card products, regions, and customer segments.
- Platt calibration assumes the relationship between model margins and outcome remains stable. Production use would require ongoing calibration, drift, and subgroup monitoring.
- This is a research and portfolio demonstration, not an autonomous payment authorization system.
