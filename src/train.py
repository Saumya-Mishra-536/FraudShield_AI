"""Train and compare symmetric and asymmetric fraud models.

Run from the repository root:
    python -m src.train

The held-out test set is evaluated once after threshold selection on validation data.
Cost values are illustrative defaults; replace them with institution-specific estimates
before treating reported monetary costs as operational estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.load_kaggle_data import get_dataset
from src.loss_functions import DEFAULT_C_FP, DEFAULT_C_FN, asymmetric_cost
from src.model import AsymmetricModel, SymmetricModel
from src.preprocessing import preprocess


def evaluate(model, X_test, y_test, c_fp: float, c_fn: float) -> dict:
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    predictions = (probabilities >= model.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    total_cost = asymmetric_cost(y_test, predictions, c_fp, c_fn)
    return {
        "threshold": float(model.threshold),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "total_asymmetric_cost": float(total_cost),
        "cost_per_transaction": float(total_cost / len(y_test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FraudShield models and compare costs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--c-fp", type=float, default=DEFAULT_C_FP)
    parser.add_argument("--c-fn", type=float, default=DEFAULT_C_FN)
    parser.add_argument("--use-smote", action="store_true",
                        help="Apply SMOTE to the training split only.")
    args = parser.parse_args()

    if args.c_fp <= 0 or args.c_fn <= 0:
        parser.error("--c-fp and --c-fn must both be greater than zero.")
    if not 0 < args.test_size < 1 or not 0 < args.val_size < 1 - args.test_size:
        parser.error("Require 0 < test-size < 1 and 0 < val-size < 1 - test-size.")

    model_dir = os.path.join(ROOT, "models")
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw = get_dataset()
    prepared = preprocess(
        raw, test_size=args.test_size, val_size=args.val_size,
        apply_smote=args.use_smote, random_state=args.seed,
        scaler_path=os.path.join(model_dir, "scaler.pkl"),
    )
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_val, y_val = prepared["X_val"], prepared["y_val"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    feature_names = prepared["feature_names"]

    print("\n[Train] Training symmetric baseline...")
    symmetric = SymmetricModel(threshold=0.5)
    symmetric.fit(X_train, y_train, X_val, y_val, feature_names, class_counts)
    symmetric.tune_threshold(X_val, y_val, args.c_fp, args.c_fn)

    print("\n[Train] Training asymmetric cost-sensitive model...")
    asymmetric = AsymmetricModel(c_fp=args.c_fp, c_fn=args.c_fn)
    asymmetric.fit(X_train, y_train, X_val, y_val, feature_names, class_counts)

    symmetric.save(os.path.join(model_dir, "symmetric_model.pkl"))
    asymmetric.save(os.path.join(model_dir, "asymmetric_model.pkl"))

    # Test split is never used for threshold tuning.
    results = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "Kaggle Credit Card Fraud Detection (MLG-ULB)",
            "rows_loaded": int(len(raw)),
            "fraud_rows_loaded": int(raw["Class"].sum()),
            "random_seed": args.seed,
            "test_size": args.test_size,
            "validation_size": args.val_size,
            "smote_enabled": bool(args.use_smote),
            "false_positive_cost_assumption": args.c_fp,
            "false_negative_cost_assumption": args.c_fn,
            "cost_note": "Illustrative cost units, not validated banking loss estimates.",
            "feature_names": feature_names,
        },
        "symmetric_model": evaluate(symmetric, X_test, y_test, args.c_fp, args.c_fn),
        "asymmetric_model": evaluate(asymmetric, X_test, y_test, args.c_fp, args.c_fn),
    }
    metrics_path = os.path.join(output_dir, "model_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== Held-out test results ===")
    for name in ("symmetric_model", "asymmetric_model"):
        print(f"\n{name.replace('_', ' ').title()}")
        for key, value in results[name].items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\n[Train] Metrics written to {metrics_path}")
    print("[Train] Models and scaler saved under models/.")


if __name__ == "__main__":
    main()
