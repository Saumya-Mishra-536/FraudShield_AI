"""
Kaggle Credit Card Fraud Dataset Loader
========================================
Dataset: Credit Card Fraud Detection — MLG-ULB (Université Libre de Bruxelles)
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""
import os
import warnings

import numpy as np
import pandas as pd

EXPECTED_ROWS = 284807
EXPECTED_FRAUD = 492
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_validate(path: str) -> pd.DataFrame:
    """Load a CSV or gzip-compressed CSV and validate its expected schema."""
    print(f"[Loader] Reading: {path}")
    df = pd.read_csv(path, compression="gzip" if path.lower().endswith(".gz") else None)
    expected_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing = [column for column in expected_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if df.isnull().values.any():
        null_count = int(df.isnull().sum().sum())
        raise ValueError(
            f"Dataset contains {null_count} missing values. "
            "Resolve missing values explicitly before training."
        )

    non_numeric = [
        column for column in expected_cols
        if not pd.api.types.is_numeric_dtype(df[column].dtype)
        or pd.api.types.is_bool_dtype(df[column].dtype)
    ]
    if non_numeric:
        raise ValueError(f"Expected numeric values in columns: {non_numeric}")

    numeric_values = df[expected_cols].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        bad_columns = [
            column for column in expected_cols
            if not np.isfinite(df[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"Dataset contains infinite or non-finite values in columns: {bad_columns}")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only binary labels 0 and 1.")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount contains negative values.")
    if (df["Time"] < 0).any():
        raise ValueError("Time contains negative values.")

    pca_cols = [f"V{i}" for i in range(1, 29)]
    shifted_features = []
    for column in pca_cols:
        if float(df[column].abs().quantile(0.999)) > 100:
            shifted_features.append(column)
        if int(df[column].nunique(dropna=False)) <= 1:
            shifted_features.append(f"{column} (constant)")
    if shifted_features:
        warnings.warn(
            "PCA feature distribution is unusual for the reference dataset in: "
            + ", ".join(shifted_features),
            RuntimeWarning,
            stacklevel=2,
        )
    if float(df["Time"].max()) > 172_800:
        warnings.warn(
            "Time exceeds the roughly 48-hour span of the reference dataset; review the feature time basis.",
            RuntimeWarning,
            stacklevel=2,
        )
    if float(df["Amount"].max()) > 1_000_000:
        warnings.warn(
            "Amount has an unusually large maximum for the reference dataset; review units and input distribution.",
            RuntimeWarning,
            stacklevel=2,
        )

    fraud_count = int(df["Class"].sum())
    print(f"[Loader] Loaded {len(df):,} transactions")
    print(f"[Loader] Legitimate: {len(df) - fraud_count:,}")
    print(f"[Loader] Fraudulent: {fraud_count:,} ({df['Class'].mean():.4%})")
    print(
        f"[Loader] Time range: {df['Time'].min():.0f}s–{df['Time'].max():.0f}s; "
        f"amount range: {df['Amount'].min():.2f}–{df['Amount'].max():,.2f}"
    )
    if len(df) != EXPECTED_ROWS:
        print(f"[Loader] WARNING: expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    if fraud_count != EXPECTED_FRAUD:
        print(f"[Loader] WARNING: expected {EXPECTED_FRAUD} fraud cases, found {fraud_count}")
    return df


def get_dataset(gz_path: str = None, csv_path: str = None) -> pd.DataFrame:
    """Load an explicit path or auto-detect common dataset filenames."""
    candidates = [p for p in (gz_path, csv_path) if p]
    candidates.extend([
        os.path.join(DATA_DIR, "creditcard.csv.gz"),
        os.path.join(DATA_DIR, "creditcard.csv..gz"),
        os.path.join(DATA_DIR, "creditcard.csv"),
    ])
    for path in candidates:
        if os.path.isfile(path):
            return load_and_validate(path)

    raise FileNotFoundError(
        "Credit card dataset not found. Download it from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place "
        f"creditcard.csv or creditcard.csv.gz in {DATA_DIR}."
    )


if __name__ == "__main__":
    dataset = get_dataset()
    print(dataset[["Time", "Amount", "Class"]].head().to_string())
