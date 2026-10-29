"""
Fraud Detection Models
======================
Compares a standard XGBoost baseline with a cost-sensitive XGBoost model.
"""
import os
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb

from src.loss_functions import (
    DEFAULT_C_FP, DEFAULT_C_FN, optimal_threshold,
    xgboost_asymmetric_objective,
)


class BaseFraudModel:
    """Shared prediction, threshold-tuning, and artifact persistence interface."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.feature_names = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def tune_threshold(self, X_val, y_val, c_fp=DEFAULT_C_FP, c_fn=DEFAULT_C_FN) -> float:
        probabilities = self.predict_proba(X_val)
        threshold, _, _, _ = optimal_threshold(y_val, probabilities, c_fp, c_fn)
        self.threshold = float(threshold)
        print(f"[{self.__class__.__name__}] Validation threshold: {self.threshold:.3f}")
        return self.threshold

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(
            {"model": self.model, "threshold": self.threshold,
             "feature_names": self.feature_names},
            path,
        )
        print(f"[{self.__class__.__name__}] Saved to {path}")

    def load(self, path: str):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.feature_names = bundle["feature_names"]
        print(f"[{self.__class__.__name__}] Loaded from {path}")


class SymmetricModel(BaseFraudModel):
    """Baseline XGBoost classifier using standard logistic loss."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        self.name = "Symmetric XGBoost"

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None, class_counts=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        counts = class_counts or {
            0: int(np.sum(np.asarray(y_train) == 0)),
            1: int(np.sum(np.asarray(y_train) == 1)),
        }
        # Baseline class-imbalance weighting; cost asymmetry is handled at threshold selection.
        scale = counts[0] / max(counts[1], 1)
        self.model = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=500,
            learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale,
            eval_metric="aucpr", early_stopping_rounds=30,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"[SymmetricModel] Best iteration: {self.model.best_iteration}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class AsymmetricModel(BaseFraudModel):
    """
    Cost-sensitive XGBoost using a custom weighted logistic objective.

    The objective directly weights false-negative and false-positive losses.
    No extra class-imbalance sample weights are applied here, avoiding counting
    cost/imbalance weighting more than once.
    """

    def __init__(self, c_fp: float = DEFAULT_C_FP, c_fn: float = DEFAULT_C_FN,
                 threshold: Optional[float] = None):
        if c_fp <= 0 or c_fn <= 0:
            raise ValueError("c_fp and c_fn must both be positive.")
        super().__init__(threshold if threshold is not None else c_fp / (c_fp + c_fn))
        self.c_fp = float(c_fp)
        self.c_fn = float(c_fn)
        self.name = "Asymmetric XGBoost"

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None, class_counts=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        params = {
            "max_depth": 6, "eta": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": 42, "nthread": -1,
            "disable_default_eval_metric": True,
        }
        objective = xgboost_asymmetric_objective(self.c_fp, self.c_fn)

        def aucpr_eval(raw_scores, dmatrix):
            from sklearn.metrics import average_precision_score
            labels = dmatrix.get_label()
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -40, 40)))
            return "aucpr", float(average_precision_score(labels, probabilities))

        evals_result = {}
        self.model = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=500,
            obj=objective, custom_metric=aucpr_eval,
            evals=[(dval, "val")], early_stopping_rounds=30,
            evals_result=evals_result, verbose_eval=False,
        )
        self._evals_result = evals_result
        print(f"[AsymmetricModel] Best iteration: {self.model.best_iteration}")
        self.tune_threshold(X_val, y_val, self.c_fp, self.c_fn)

    def predict_proba(self, X):
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        best_iteration = getattr(self.model, "best_iteration", None)
        if best_iteration is not None:
            raw_scores = self.model.predict(dmatrix, iteration_range=(0, best_iteration + 1))
        else:
            raw_scores = self.model.predict(dmatrix)
        return 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -40, 40)))

    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        scores = self.model.get_score(importance_type=importance_type)
        return {feature: float(scores.get(feature, 0.0))
                for feature in (self.feature_names or [])}
