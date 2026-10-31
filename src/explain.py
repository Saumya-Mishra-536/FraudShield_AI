"""
Explainability Module — Local SHAP Attribution & Global Importance
===================================================================
Provides exact local feature attributions using SHAP TreeExplainer on XGBoost,
strictly distinguishing between:
1. Local Explanation: Why did *this specific transaction* receive its risk score?
   Attributed via additive SHAP values: logit(p) = base_value + sum(shap_values)
2. Global Feature Importance: Which features matter most across the dataset overall?
   Computed via average gain or mean |SHAP| over background transactions.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
import shap

from src.preprocessing import FEATURE_COLS


class TransactionExplainer:
    """
    Local and global explainability provider for XGBoost fraud detection models.
    Uses SHAP TreeExplainer (exact tree traversal algorithm).
    """

    def __init__(self, booster: xgb.Booster, feature_names: Optional[List[str]] = None):
        self.booster = booster
        self.feature_names = feature_names or FEATURE_COLS
        self.explainer = shap.TreeExplainer(self.booster)
        self.base_value = float(
            self.explainer.expected_value
            if np.isscalar(self.explainer.expected_value)
            else self.explainer.expected_value[0]
        )
        self.base_probability = float(1.0 / (1.0 + np.exp(-np.clip(self.base_value, -40, 40))))

    def explain_instance(
        self,
        X_scaled: np.ndarray,
        raw_feature_dict: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute local SHAP attribution for a single transaction.

        Args:
            X_scaled: 2D numpy array of shape (1, n_features) preprocessed features.
            raw_feature_dict: Optional dictionary of original unscaled feature values for human display.
            top_k: Number of top driving features to highlight.

        Returns:
            Dictionary containing local SHAP values, risk-increasing factors,
            risk-decreasing factors, base probability, and final probability.
        """
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        dmat = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        shap_values = self.explainer.shap_values(dmat)[0]

        total_margin = float(self.base_value + np.sum(shap_values))
        reconstructed_prob = float(1.0 / (1.0 + np.exp(-np.clip(total_margin, -40, 40))))

        items = []
        for i, fname in enumerate(self.feature_names):
            phi = float(shap_values[i])
            val_scaled = float(X_scaled[0, i])
            raw_val = (
                float(raw_feature_dict.get(fname, val_scaled))
                if raw_feature_dict
                else val_scaled
            )
            items.append({
                "feature": fname,
                "shap_value": round(phi, 5),
                "scaled_value": round(val_scaled, 4),
                "raw_value": round(raw_val, 4) if raw_val is not None else None,
                "effect": "increases_fraud_risk" if phi > 0 else "decreases_fraud_risk",
                "abs_impact": abs(phi),
            })

        # Separate risk-increasing (red flags) vs risk-decreasing (signals of legitimacy)
        risk_increasing = [item for item in items if item["shap_value"] > 0]
        risk_increasing.sort(key=lambda x: x["shap_value"], reverse=True)

        risk_decreasing = [item for item in items if item["shap_value"] < 0]
        risk_decreasing.sort(key=lambda x: x["shap_value"])  # most negative first

        # Top overall local drivers by absolute impact
        all_sorted = sorted(items, key=lambda x: x["abs_impact"], reverse=True)

        return {
            "explanation_type": "local_shap_attribution",
            "base_value_log_odds": round(self.base_value, 4),
            "base_probability": round(self.base_probability, 6),
            "total_log_odds": round(total_margin, 4),
            "fraud_probability": round(reconstructed_prob, 6),
            "top_risk_increasing_factors": risk_increasing[:top_k],
            "top_risk_decreasing_factors": risk_decreasing[:top_k],
            "top_drivers_by_magnitude": all_sorted[:top_k],
            "all_feature_attributions": items,
        }

    def get_global_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Return model-level global feature importance from XGBoost tree splits.

        Args:
            importance_type: 'gain', 'weight' (frequency), or 'cover'.
            top_k: Number of top features to return.
        """
        scores = self.booster.get_score(importance_type=importance_type)
        total = sum(scores.values()) if scores else 1.0

        ranking = [
            {
                "feature": feat,
                "importance_score": round(float(scores.get(feat, 0.0)), 4),
                "relative_importance": round(float(scores.get(feat, 0.0)) / total, 4),
            }
            for feat in self.feature_names
        ]
        ranking.sort(key=lambda x: x["importance_score"], reverse=True)

        return {
            "explanation_type": f"global_{importance_type}_importance",
            "description": f"Global feature ranking across all training trees based on split {importance_type}.",
            "top_features": ranking[:top_k],
            "full_ranking": ranking,
        }

