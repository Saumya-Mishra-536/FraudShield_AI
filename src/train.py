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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.load_kaggle_data import get_dataset
from src.calibration import bootstrap_brier_interval, grouped_calibration_report
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
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "probability_calibrated": bool(getattr(model, "calibrator", None) is not None),
        "calibration_method": getattr(model, "calibration_method", None),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "total_asymmetric_cost": float(total_cost),
        "cost_per_transaction": float(total_cost / len(y_test)),
    }



def plot_calibration_curve(y_true, probabilities, output_path: str) -> None:
    """Save a held-out reliability diagram and its Brier score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(predicted, observed, "o-", color="#2563eb", label=f"Asymmetric model (Brier={brier:.4f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud frequency",
           title="Held-out calibration curve")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FraudShield models and compare costs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--c-fp", type=float, default=DEFAULT_C_FP)
    parser.add_argument("--c-fn", type=float, default=DEFAULT_C_FN)
    parser.add_argument("--use-smote", action="store_true",
                        help="Apply SMOTE to the training split only.")
    parser.add_argument(
        "--split-strategy",
        choices=("random", "temporal"),
        default="random",
        help="Use a stratified random split or chronological train/validation/test windows.",
    )
    args = parser.parse_args()

    if args.c_fp <= 0 or args.c_fn <= 0:
        parser.error("--c-fp and --c-fn must both be greater than zero.")
    if not 0 < args.test_size < 1 or not 0 < args.val_size < 1 - args.test_size:
        parser.error("Require 0 < test-size < 1 and 0 < val-size < 1 - test-size.")

    if args.split_strategy == "temporal":
        model_dir = os.path.join(ROOT, "models", "time_holdout")
        output_dir = os.path.join(ROOT, "outputs", "time_holdout")
    else:
        model_dir = os.path.join(ROOT, "models")
        output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw = get_dataset()
    prepared = preprocess(
        raw, test_size=args.test_size, val_size=args.val_size,
        apply_smote=args.use_smote, random_state=args.seed,
        scaler_path=os.path.join(model_dir, "scaler.pkl"),
        split_strategy=args.split_strategy,
    )
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_val, y_val = prepared["X_val"], prepared["y_val"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    split_class_counts = {
        split_name: {
            int(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }
        for split_name, labels in (
            ("train", y_train), ("validation", y_val), ("test", y_test)
        )
    }
    feature_names = prepared["feature_names"]

    print("\n[Train] Training symmetric baseline...")
    symmetric = SymmetricModel(threshold=0.5)
    symmetric.fit(X_train, y_train, X_val, y_val, feature_names, class_counts)
    symmetric.tune_threshold(X_val, y_val, args.c_fp, args.c_fn)

    print("\n[Train] Training asymmetric cost-sensitive model...")
    asymmetric = AsymmetricModel(c_fp=args.c_fp, c_fn=args.c_fn)
    asymmetric.fit(X_train, y_train, X_val, y_val, feature_names, class_counts)

    # Save model artifacts
    symmetric.save(os.path.join(model_dir, "symmetric_model.pkl"))
    asymmetric.save(os.path.join(model_dir, "asymmetric_model.pkl"))

    # Persist test predictions for lightweight, dynamic dashboard simulation
    sym_proba = symmetric.predict_proba(X_test).astype(np.float32)
    asym_proba = asymmetric.predict_proba(X_test).astype(np.float32)
    np.savez_compressed(
        os.path.join(output_dir, "test_predictions.npz"),
        y_test=y_test.astype(np.int8),
        sym_proba=sym_proba,
        asym_proba=asym_proba,
    )

    # The held-out split is used only for final evaluation and descriptive uncertainty.
    test_indices = prepared["test_indices"]
    if args.split_strategy == "temporal":
        split_time_ranges = {
            split_name: {
                "start_seconds": float(raw.iloc[indices]["Time"].min()),
                "end_seconds": float(raw.iloc[indices]["Time"].max()),
            }
            for split_name, indices in (
                ("train", prepared["train_indices"]),
                ("validation", prepared["validation_indices"]),
                ("test", test_indices),
            )
        }
    else:
        split_time_ranges = None
    calibration_evaluation = {
        "brier_score_bootstrap": bootstrap_brier_interval(
            y_test, asym_proba, split_strategy=args.split_strategy
        ),
        "held_out_groups": grouped_calibration_report(
            y_test,
            asym_proba,
            raw.iloc[test_indices]["Amount"].to_numpy(),
            raw.iloc[test_indices]["Time"].to_numpy(),
            split_strategy=args.split_strategy,
        ),
    }
    results = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "Kaggle Credit Card Fraud Detection (MLG-ULB)",
            "rows_loaded": int(len(raw)),
            "fraud_rows_loaded": int(raw["Class"].sum()),
            "random_seed": args.seed,
            "test_size": args.test_size,
            "validation_size": args.val_size,
            "split_strategy": args.split_strategy,
            "split_time_ranges": split_time_ranges,
            "split_class_counts": split_class_counts,
            "smote_enabled": bool(args.use_smote),
            "false_positive_cost_assumption": args.c_fp,
            "false_negative_cost_assumption": args.c_fn,
            "cost_note": "Illustrative cost units, not validated banking loss estimates.",
            "threshold_selection_method": asymmetric.threshold_selection_method,
            "feature_names": feature_names,
        },
        "symmetric_model": evaluate(symmetric, X_test, y_test, args.c_fp, args.c_fn),
        "asymmetric_model": evaluate(asymmetric, X_test, y_test, args.c_fp, args.c_fn),
        "calibration_evaluation": calibration_evaluation,
    }
    plot_calibration_curve(y_test, asym_proba, os.path.join(output_dir, "plots", "calibration_curve.png"))
    metrics_path = os.path.join(output_dir, "model_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    calibration_path = os.path.join(output_dir, "calibration_evaluation.json")
    with open(calibration_path, "w", encoding="utf-8") as file:
        json.dump(calibration_evaluation, file, indent=2)

    print("\n=== Held-out test results ===")
    for name in ("symmetric_model", "asymmetric_model"):
        print(f"\n{name.replace('_', ' ').title()}")
        for key, value in results[name].items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\n[Train] Metrics written to {metrics_path}")
    print(f"[Train] Models and scaler saved under {model_dir}.")


if __name__ == "__main__":
    main()
