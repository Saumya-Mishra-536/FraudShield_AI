# FraudShield AI — Cost-Sensitive Credit Card Fraud Detection

FraudShield AI explores a practical fraud-detection trade-off: missing a fraudulent transaction and incorrectly flagging a legitimate one do not necessarily have equal costs. It compares a standard XGBoost baseline with a cost-sensitive XGBoost model and evaluates both using precision, recall, PR-AUC, ROC-AUC, confusion-matrix counts, and an explicit cost function.

> **Research/portfolio project.** The default false-positive and false-negative costs are illustrative assumptions, not estimates validated by a bank. Model scores are not a guarantee of fraud, and this repository is not a production payment-blocking system.

## Problem

Fraudulent transactions are rare compared with legitimate ones in the Kaggle Credit Card Fraud Detection dataset. Accuracy alone can hide poor fraud detection. FraudShield focuses on:
- detecting fraud while making the false-positive/false-negative trade-off visible;
- comparing a symmetric baseline with a cost-sensitive model;
- selecting decision thresholds on validation data rather than the final test set;
- reporting classification metrics and estimated cost under configurable assumptions.

## Method

1. Load and validate the Kaggle dataset.
2. Engineer `Amount_log`, `Time_sin`, and `Time_cos`; preserve the anonymized `V1`–`V28` inputs.
3. Create stratified train, validation, and test splits.
4. Fit preprocessing only on the training split. Optional SMOTE is applied only to training data.
5. Train a symmetric XGBoost baseline and a cost-sensitive XGBoost model.
6. Select decision thresholds on validation data.
7. Evaluate each model once on the held-out test split and save metrics as JSON.

### Cost function

`Total cost = C_FP × FP + C_FN × FN`

- `C_FP`: assumed cost of flagging a legitimate transaction.
- `C_FN`: assumed cost of missing a fraudulent transaction.

The defaults are `C_FP = 5` and `C_FN = 20` in illustrative cost units. Change them to test scenarios; do not interpret them as validated real-world currency costs.

## Repository structure

```text
data/
  creditcard.csv..gz       # Compressed dataset currently in this repository
  load_kaggle_data.py      # Dataset loading and validation
src/
  preprocessing.py         # Feature engineering, splits, scaling, optional SMOTE
  loss_functions.py        # Cost function, threshold selection, custom objective
  model.py                 # Symmetric and asymmetric XGBoost models
  train.py                 # Reproducible training and held-out evaluation
  evaluate.py              # Additional visual analysis utilities
  api.py                   # FastAPI prediction endpoints
tests/
  test_core.py             # Unit tests for core utilities
models/                    # Created locally by training; model artifacts
outputs/                   # Created locally; evaluation JSON and plots
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Saumya-Mishra-536/FraudShield_AI.git
cd FraudShield_AI

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset is large. If you use a fresh clone or remove the committed archive, download the Credit Card Fraud Detection dataset from [Kaggle (MLG-ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` or a supported compressed CSV in `data/`.

## Train and evaluate

Run from the repository root:

```bash
python -m src.train
```

Optional flags:

```bash
python -m src.train --c-fp 5 --c-fn 20 --seed 42
python -m src.train --use-smote
```

The run creates:
- `models/scaler.pkl`
- `models/symmetric_model.pkl`
- `models/asymmetric_model.pkl`
- `outputs/model_comparison.json`

Threshold selection uses the validation split. The test split is reserved for final reporting. Training may take several minutes depending on hardware.

## Run the API

Train the model first, then:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The API accepts the dataset's anonymized PCA features plus `Amount` and `Time`. These inputs are specific to this dataset and are **not** a generic transaction form for real card systems.

## Tests

```bash
pytest tests/ -v
```

## Limitations and responsible interpretation

- The dataset covers a short historical window and uses anonymized PCA features; it does not represent every bank, country, payment channel, or current fraud pattern.
- The default cost values are assumptions. Results can change materially when cost assumptions or decision thresholds change.
- A lower total cost on this dataset does not establish production readiness or prove that the asymmetric model is universally better.
- Probability calibration, temporal/generalization tests, drift monitoring, model-card documentation, and a deployment security review are future work.
- Do not use model predictions alone to automatically deny real transactions.

## Data source

Andrea Dal Pozzolo et al., *Credit Card Fraud Detection*, available via the [Kaggle MLG-ULB dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Respect the dataset's terms of use and cite the dataset in downstream work.
