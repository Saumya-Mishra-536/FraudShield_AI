"""
FraudShield AI — Interactive Research Dashboard
================================================
A research & portfolio demonstration comparing symmetric vs asymmetric cost-sensitive
fraud detection models on the Kaggle Credit Card Fraud dataset.

Run with:
    streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure silent Matplotlib config directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.calibration import model_output_to_score
from src.decision_engine import DecisionEngine
from src.explain import TransactionExplainer, generate_shap_narrative
from src.loss_functions import DEFAULT_C_FN, DEFAULT_C_FP, asymmetric_cost
from src.preprocessing import FEATURE_COLS, engineer_features

# Page Configuration
st.set_page_config(
    page_title="FraudShield AI — Cost-Sensitive Fraud Architecture",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #f8fafc; }
    .main-header { font-size: 2.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #475569; margin-bottom: 1.2rem; }
    @media (prefers-color-scheme: dark) {
        .main-header { color: #f8fafc; }
        .sub-header { color: #cbd5e1; }
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #991b1b;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1.0rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .badge-allow { background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-challenge { background-color: #fef3c7; color: #92400e; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-block { background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-hard-block { background-color: #7f1d1d; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    """Load pre-trained models, scaler, metrics, explainer, and test predictions."""
    models_dir = os.getenv("MODEL_DIR", os.path.join(ROOT, "models"))
    outputs_dir = os.getenv("OUTPUT_DIR", os.path.join(ROOT, "outputs"))

    sym_path = os.path.join(models_dir, "symmetric_model.pkl")
    asym_path = os.path.join(models_dir, "asymmetric_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    metrics_path = os.path.join(outputs_dir, "model_comparison.json")
    test_pred_path = os.path.join(outputs_dir, "test_predictions.npz")

    sym_bundle = joblib.load(sym_path) if os.path.exists(sym_path) else None
    asym_bundle = joblib.load(asym_path) if os.path.exists(asym_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    test_preds = None
    if os.path.exists(test_pred_path):
        test_preds = np.load(test_pred_path)

    explainer = None
    decision_engine = None
    if asym_bundle and "model" in asym_bundle:
        feat_names = asym_bundle.get("feature_names", FEATURE_COLS)
        explainer = TransactionExplainer(asym_bundle["model"], feat_names)
        calibrated_t = float(asym_bundle.get("threshold", 0.160))
        decision_engine = DecisionEngine(fraud_threshold=calibrated_t)

    return sym_bundle, asym_bundle, scaler, metrics, explainer, decision_engine, test_preds


sym_model, asym_model, scaler, metrics, explainer, decision_engine, test_preds = load_all_artifacts()

# Session State for prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# Header
st.markdown('<div class="main-header">🛡️ FraudShield AI: Cost-Sensitive Fraud Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asymmetric Gradient Boosting with Kahneman-Tversky Loss Weighting & Local SHAP Explainability</div>', unsafe_allow_html=True)

# Mandatory Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Research & Portfolio Demonstration:</strong>
    This interactive dashboard demonstrates cost-sensitive machine learning, threshold optimization,
    and instance-level explainability on the Kaggle MLG-ULB credit card dataset (284,807 transactions).
    Reported monetary values use illustrative cost assumptions (USD 5 per false positive and USD 20 per false negative),
    not validated banking loss parameters. It is an educational prototype, not an active financial fraud prevention system.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Status-Verified%20Pipeline-success?style=for-the-badge")
    st.title("Navigation & Parameters")
    tab_choice = st.radio(
        "Select Module:",
        [
            "📊 Model Comparison & Audit",
            "🎚️ Decision Threshold & Cost Simulator",
            "🔍 Live Transaction Scoring & SHAP",
            "📐 Mathematical & Behavioral Theory",
        ],
    )
    st.divider()
    st.markdown("### Default Cost Assumptions")
    st.markdown("- **False Positive ($C_{FP}$)**: $5.00  \n  *(Customer friction, verification alert, card re-issue risk)*")
    st.markdown("- **False Negative ($C_{FN}$)**: $20.00  \n  *(Chargeback fee, investigation cost, direct stolen balance loss)*")
    st.markdown("- **Asymmetry Ratio ($\\lambda$)**: 4.0×  \n  *(Tversky-Kahneman loss aversion multiplier)*")
    st.divider()
    if decision_engine:
        st.markdown("### Calibrated Decision Engine")
        st.markdown(f"- **Fraud Cutoff ($\\tau$)**: `{decision_engine.threshold:.3f}`")
        st.markdown(f"- **Challenge Tier ($\\tau_{{c}}$)**: `{decision_engine.threshold_challenge:.3f}`")
        st.markdown(f"- **Hard Block Tier ($\\tau_{{hb}}$)**: `{decision_engine.threshold_hard_block:.3f}`")
    st.divider()
    st.caption("FraudShield AI · Built with XGBoost 2.x, SHAP, and Streamlit")

# ==============================================================================
# TAB 1: Model Comparison & Audit
# ==============================================================================
if tab_choice == "📊 Model Comparison & Audit":
    test_row_count = sum(
        int(metrics["asymmetric_model"][key])
        for key in ("true_positives", "false_positives", "false_negatives", "true_negatives")
    ) if metrics else 0
    st.subheader(f"Audited Test Evaluation on Untouched Held-Out Split ({test_row_count:,} Transactions)")
    st.write(
        "Both models were trained on 199,364 transactions (0.173% positive rate), tuned on a distinct validation split (28,481 rows), "
        "and evaluated **strictly once** on the held-out test set."
    )

    if metrics:
        s_m = metrics["symmetric_model"]
        a_m = metrics["asymmetric_model"]

        c1, c2, c3, c4 = st.columns(4)
        cost_diff = s_m["total_asymmetric_cost"] - a_m["total_asymmetric_cost"]
        cost_pct = (cost_diff / s_m["total_asymmetric_cost"]) * 100

        with c1:
            st.metric("Total Asymmetric Cost", f"${a_m['total_asymmetric_cost']:,.0f}", f"-${cost_diff:,.0f} ({cost_pct:.1f}%)", delta_color="inverse")
        with c2:
            precision_delta_pp = (a_m["precision"] - s_m["precision"]) * 100
            st.metric("Test Precision", f"{a_m['precision']:.2%}", f"{precision_delta_pp:+.2f} pp")
        with c3:
            recall_delta_pp = (a_m["recall"] - s_m["recall"]) * 100
            st.metric("Test Recall", f"{a_m['recall']:.2%}", f"{recall_delta_pp:+.2f} pp")
        with c4:
            st.metric("F1 Score", f"{a_m['f1']:.4f}", f"+{a_m['f1'] - s_m['f1']:.4f}")

        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 📋 Side-by-Side Held-Out Performance")
            comparison_df = pd.DataFrame({
                "Metric": [
                    "Optimal Decision Threshold",
                    "Precision",
                    "Recall (Fraud Caught Rate)",
                    "F1 Score",
                    "ROC-AUC",
                    "PR-AUC (Average Precision)",
                    "Brier Score",
                    "Probability Calibration",
                    "True Positives (Caught Fraud)",
                    "False Positives (False Alarms)",
                    "False Negatives (Missed Fraud)",
                    "True Negatives",
                    "Total Asymmetric Cost",
                    "Cost Per Transaction",
                ],
                "Symmetric XGBoost (Baseline)": [
                    f"{s_m['threshold']:.3f}",
                    f"{s_m['precision']:.4f}",
                    f"{s_m['recall']:.4f}",
                    f"{s_m['f1']:.4f}",
                    f"{s_m['roc_auc']:.4f}",
                    f"{s_m['average_precision_pr_auc']:.4f}",
                    f"{s_m['brier_score']:.6f}",
                    "Calibrated" if s_m.get("probability_calibrated") else "Not calibrated",
                    f"{s_m['true_positives']}",
                    f"{s_m['false_positives']}",
                    f"{s_m['false_negatives']}",
                    f"{s_m['true_negatives']:,}",
                    f"${s_m['total_asymmetric_cost']:,.2f}",
                    f"${s_m['cost_per_transaction']:.5f}",
                ],
                "Asymmetric XGBoost (Cost-Sensitive)": [
                    f"{a_m['threshold']:.3f}",
                    f"{a_m['precision']:.4f}",
                    f"{a_m['recall']:.4f}",
                    f"{a_m['f1']:.4f}",
                    f"{a_m['roc_auc']:.4f}",
                    f"{a_m['average_precision_pr_auc']:.4f}",
                    f"{a_m['brier_score']:.6f}",
                    "Platt scaling" if a_m.get("calibration_method") and "platt" in a_m["calibration_method"].lower() else (
                        "Calibrated" if a_m.get("probability_calibrated") else "Not calibrated"
                    ),
                    f"{a_m['true_positives']}",
                    f"{a_m['false_positives']}",
                    f"{a_m['false_negatives']}",
                    f"{a_m['true_negatives']:,}",
                    f"${a_m['total_asymmetric_cost']:,.2f}",
                    f"${a_m['cost_per_transaction']:.5f}",
                ],
            })
            st.dataframe(comparison_df, hide_index=True)

        with col_right:
            st.markdown("### 🎯 Key Evaluation Takeaways")
            symmetric_positives = s_m["true_positives"] + s_m["false_negatives"]
            asymmetric_positives = a_m["true_positives"] + a_m["false_negatives"]
            fp_reduction = s_m["false_positives"] - a_m["false_positives"]
            fp_reduction_pct = (
                fp_reduction / s_m["false_positives"] * 100
                if s_m["false_positives"]
                else 0.0
            )
            theoretical_bayes_threshold = DEFAULT_C_FP / (DEFAULT_C_FP + DEFAULT_C_FN)
            st.info(
                f"""
                1. **False Alarms:** False positives changed from **{s_m['false_positives']:,}** to **{a_m['false_positives']:,}** ({fp_reduction_pct:.1f}% fewer under this test threshold).
                2. **Fraud Detection:** The symmetric model caught **{s_m['true_positives']:,} of {symmetric_positives:,}** cases; the asymmetric model caught **{a_m['true_positives']:,} of {asymmetric_positives:,}**. Recall changed from **{s_m['recall']:.2%}** to **{a_m['recall']:.2%}**.
                3. **Illustrative Cost:** Estimated cost under the stated assumptions changed from **USD {s_m['total_asymmetric_cost']:,.0f}** to **USD {a_m['total_asymmetric_cost']:,.0f}** on this held-out test set.
                4. **Thresholds:** Validation-selected thresholds were **{s_m['threshold']:.3f}** (symmetric) and **{a_m['threshold']:.3f}** (asymmetric). The simple Bayes reference under the illustrative costs is **{theoretical_bayes_threshold:.3f}**; it is a reference point, not a substitute for validation.
                """
            )

        st.markdown("---")
        st.subheader("🖼️ Generated Evaluation Artifacts")
        plot_tabs = st.tabs(["Confusion Matrices", "PR & ROC Curves", "Threshold Tuning Curves", "SHAP Feature Summary"])

        plots_dir = os.path.join(ROOT, "outputs", "plots")
        with plot_tabs[0]:
            cm_img = os.path.join(plots_dir, "confusion_matrix_comparison.png")
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Confusion Matrices under illustrative cost weights")
            else:
                st.warning("Confusion matrix plot not found. Run python -m src.evaluate to regenerate.")

        with plot_tabs[1]:
            roc_pr_img = os.path.join(plots_dir, "roc_pr_curves.png")
            if os.path.exists(roc_pr_img):
                st.image(roc_pr_img, caption="ROC and Precision-Recall Curves")

        with plot_tabs[2]:
            tuning_img = os.path.join(plots_dir, "threshold_tuning.png")
            if os.path.exists(tuning_img):
                st.image(tuning_img, caption="Empirical vs Theoretical Cost Minimization Curves")

        with plot_tabs[3]:
            shap_img = os.path.join(plots_dir, "shap_summary.png")
            if os.path.exists(shap_img):
                st.image(shap_img, caption="Global SHAP Summary across test sample")

    else:
        st.warning("No model comparison file found. Please run: python -m src.train")


# ==============================================================================
# TAB 2: Decision Threshold & Cost Simulator (100% Dynamic on Real Test Set)
# ==============================================================================
elif tab_choice == "🎚️ Decision Threshold & Cost Simulator":
    st.subheader("Interactive Decision Threshold & Economic Cost Simulator")
    st.write(
        "Dynamically evaluate the trade-off between customer friction ($FP$) and financial losses ($FN$) "
        "calculated directly against the **56,962 held-out test transactions**."
    )

    if test_preds is None:
        st.error("Test prediction cache not found (`outputs/test_predictions.npz`). Please run: `python -m src.train`")
    else:
        y_test = test_preds["y_test"]
        asym_proba = test_preds["asym_proba"]
        total_test_rows = len(y_test)
        total_frauds = int(np.sum(y_test == 1))
        total_legits = int(np.sum(y_test == 0))

        col_sim_l, col_sim_r = st.columns([1, 2])

        with col_sim_l:
            st.markdown("#### 1. Operational Cost Assumptions")
            sim_c_fp = st.slider("Cost per False Positive ($C_{FP}$)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
            sim_c_fn = st.slider("Cost per False Negative ($C_{FN}$)", min_value=10.0, max_value=250.0, value=20.0, step=5.0)

            theoretical_tau = sim_c_fp / (sim_c_fp + sim_c_fn)
            st.markdown(f"**Theoretical Bayes Optimal Threshold ($\\tau^*$):** `{theoretical_tau:.3f}`")

            st.markdown("#### 2. Active Decision Threshold")
            calibrated_t = decision_engine.threshold if decision_engine else 0.160
            sim_threshold = st.slider(
                "Select Decision Threshold ($\\tau$)",
                min_value=0.01,
                max_value=0.99,
                value=float(calibrated_t),
                step=0.01,
            )

            st.caption(
                f"Transactions with $P(\\text{{fraud}}) \\ge {sim_threshold:.2f}$ will be flagged / blocked; "
                f"transactions with $P < {sim_threshold:.2f}$ will be approved or challenged."
            )

        with col_sim_r:
            # 100% Dynamic empirical calculation across 56,962 test set rows
            preds_at_tau = (asym_proba >= sim_threshold).astype(np.int8)
            tp = int(np.sum((preds_at_tau == 1) & (y_test == 1)))
            fp = int(np.sum((preds_at_tau == 1) & (y_test == 0)))
            fn = int(np.sum((preds_at_tau == 0) & (y_test == 1)))
            tn = int(np.sum((preds_at_tau == 0) & (y_test == 0)))

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
            dynamic_cost = sim_c_fp * fp + sim_c_fn * fn

            # Dynamic Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{precision:.2%}")
            m2.metric("Recall (Capture)", f"{recall:.2%}", f"{tp}/{total_frauds} caught")
            m3.metric("F1 Score", f"{f1:.4f}")
            m4.metric("Total Incurred Cost", f"${dynamic_cost:,.2f}", f"${dynamic_cost/total_test_rows:.5f}/tx", delta_color="inverse")

            # Dynamic Cost Curve across candidate thresholds
            threshold_grid = np.linspace(0.01, 0.95, 100)
            cost_curve = []
            for t in threshold_grid:
                p_flags = asym_proba >= t
                t_fp = np.sum(p_flags & (y_test == 0))
                t_fn = np.sum((~p_flags) & (y_test == 1))
                cost_curve.append(sim_c_fp * t_fp + sim_c_fn * t_fn)

            min_cost_idx = int(np.argmin(cost_curve))
            empirical_opt_t = threshold_grid[min_cost_idx]
            min_empirical_cost = cost_curve[min_cost_idx]

            fig, (ax_curve, ax_cm) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

            # Cost trajectory
            ax_curve.plot(threshold_grid, cost_curve, color="#2563eb", linewidth=2.2, label="Actual Test Set Cost ($)")
            ax_curve.axvline(theoretical_tau, color="#f59e0b", linestyle="--", label=f"Bayes Theoretical (τ={theoretical_tau:.2f})")
            ax_curve.axvline(empirical_opt_t, color="#10b981", linestyle="-.", label=f"Empirical Minimum (τ={empirical_opt_t:.2f})")
            ax_curve.axvline(sim_threshold, color="#dc2626", linestyle=":", label=f"Selected Slider (τ={sim_threshold:.2f})")
            ax_curve.scatter([sim_threshold], [dynamic_cost], color="#dc2626", s=60, zorder=5)
            ax_curve.set_xlabel("Decision Threshold (τ)", fontsize=9)
            ax_curve.set_ylabel("Total Incurred Cost ($)", fontsize=9)
            ax_curve.set_title("Empirical Cost vs Threshold (56,962 Test Transactions)", fontsize=10, fontweight="bold")
            ax_curve.grid(alpha=0.3)
            ax_curve.legend(fontsize=8)

            # Dynamic Confusion Matrix
            cm_data = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(
                cm_data,
                annot=True,
                fmt=",d",
                cmap="Blues",
                cbar=False,
                ax=ax_cm,
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
            )
            ax_cm.set_title(f"Dynamic Confusion Matrix (τ={sim_threshold:.2f})", fontsize=10, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"**Simulation Summary**: At threshold $\\tau={sim_threshold:.2f}$, "
                f"the model clears `{tn:,}` legitimate transactions, flags `{fp}` false alarms (cost: ${fp * sim_c_fp:,.0f}), "
                f"catches `{tp}` frauds, and misses `{fn}` frauds (cost: ${fn * sim_c_fn:,.0f})."
            )


# ==============================================================================
# TAB 3: Live Transaction Scoring & SHAP
# ==============================================================================
elif tab_choice == "🔍 Live Transaction Scoring & SHAP":
    st.subheader("Live Inference & Local SHAP Attribution Engine")
    st.write(
        "Evaluate individual transaction profiles against the **unified decision engine** "
        "and inspect true instance-level SHAP attributions."
    )

    preset = st.selectbox(
        "Load Preset Transaction Scenario:",
        [
            "Scenario A: Normal Everyday Purchase (Low Risk)",
            "Scenario B: Suspicious Nighttime Card-Testing (High Risk)",
            "Scenario C: Borderline Ambiguous Case (Step-Up Challenge)",
        ],
    )

    preset_vals = {f"V{i}": 0.0 for i in range(1, 29)}
    if "Normal" in preset:
        preset_vals["Amount"] = 45.20
        preset_vals["Time"] = 43200.0  # 12:00 PM
        preset_vals["V14"] = 0.25
        preset_vals["V4"] = -0.10
    elif "Suspicious" in preset:
        preset_vals["Amount"] = 1.00  # Micro card test
        preset_vals["Time"] = 7200.0  # 2:00 AM
        preset_vals["V14"] = -7.50    # Extreme behavioral anomaly
        preset_vals["V4"] = 5.20
        preset_vals["V12"] = -6.10
        preset_vals["V10"] = -4.80
    else:
        preset_vals["Amount"] = 450.00
        preset_vals["Time"] = 82000.0  # 10:45 PM
        preset_vals["V14"] = -2.80
        preset_vals["V4"] = 2.10

    col_in_1, col_in_2, col_in_3 = st.columns(3)
    with col_in_1:
        tx_amount = st.number_input("Transaction Amount (Dataset Units)", min_value=0.0, value=float(preset_vals["Amount"]), step=10.0)
    with col_in_2:
        tx_time = st.number_input("Time Elapsed (Seconds from start)", min_value=0.0, value=float(preset_vals["Time"]), step=3600.0)
    with col_in_3:
        tx_v14 = st.number_input("V14 (Primary Behavioral Component)", value=float(preset_vals["V14"]), step=0.5)

    with st.expander("Adjust Additional PCA Features (V1–V28)"):
        v_cols = st.columns(4)
        for idx, i in enumerate(range(1, 29)):
            col_target = v_cols[idx % 4]
            with col_target:
                preset_vals[f"V{i}"] = st.number_input(f"V{i}", value=float(preset_vals.get(f"V{i}", 0.0)), step=0.5, key=f"inp_v{i}")

    preset_vals["Amount"] = tx_amount
    preset_vals["Time"] = tx_time
    preset_vals["V14"] = tx_v14

    if st.button("🚀 Evaluate Transaction Risk & Generate Explanation", type="primary"):
        if asym_model is None or scaler is None or explainer is None or decision_engine is None:
            st.error("Artifacts not loaded. Please train models first via: python -m src.train")
        else:
            with st.spinner("Processing feature engineering and calculating SHAP tree traversal..."):
                df_tx = pd.DataFrame([preset_vals])
                df_engineered = engineer_features(df_tx)
                X_raw = df_engineered[FEATURE_COLS].values
                X_scaled = scaler.transform(X_raw)

                # Inference via booster
                dmat = xgb.DMatrix(X_scaled, feature_names=FEATURE_COLS)
                booster = asym_model["model"]
                best_iteration = getattr(booster, "best_iteration", None)
                if best_iteration is not None:
                    raw_score = booster.predict(dmat, iteration_range=(0, best_iteration + 1))[0]
                else:
                    raw_score = booster.predict(dmat)[0]

                proba_score = model_output_to_score(raw_score, output_type=asym_model.get("output_type", "raw_margin"), calibrator=asym_model.get("calibrator"))

                # Unified Decision Engine Evaluation
                decision = decision_engine.evaluate(proba_score)

                # Local SHAP Explanation
                local_exp = explainer.explain_instance(X_scaled, raw_feature_dict=preset_vals, top_k=5)
                narrative = generate_shap_narrative(local_exp, decision.decision, top_k=3)

                # Append to history
                st.session_state["prediction_history"].append({
                    "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Amount": tx_amount,
                    "Probability": f"{decision.fraud_probability:.4%}",
                    "Risk Tier": decision.risk_tier,
                    "Decision": decision.decision,
                })

            st.markdown("---")
            st.markdown("### 🎯 Decision Summary")

            badge_class = {
                "ALLOW": "badge-allow",
                "CHALLENGE": "badge-challenge",
                "BLOCK": "badge-block",
                "HARD_BLOCK": "badge-hard-block",
            }.get(decision.decision, "badge-allow")

            r1, r2, r3, r4 = st.columns(4)
            calibrated = bool(asym_model.get("probability_calibrated", asym_model.get("calibrator") is not None))
            r1.metric("Calibrated Fraud Probability" if calibrated else "Model Risk Score", f"{decision.fraud_probability:.4%}")
            if not calibrated:
                st.caption("This artifact has no saved calibrator; the value is an uncalibrated model risk score, not a posterior probability.")
            r2.metric("Decision Threshold Used", f"{decision.threshold_used:.3f}")
            r3.metric("Assigned Risk Tier", decision.risk_tier)
            with r4:
                st.markdown("**Operational Action**")
                st.markdown(f'<span class="{badge_class}">{decision.decision}</span>', unsafe_allow_html=True)
                st.caption(decision.recommendation)

            # SHAP Narrative
            st.markdown("---")
            st.markdown("### 🧠 SHAP Attribution Narrative")
            for item in narrative:
                st.markdown(f"- {item}")

            # Local SHAP Feature Breakdown
            st.markdown("---")
            st.markdown("### 🔍 Exact Instance-Level SHAP Feature Contributions")
            st.caption("Additive decomposition in log-odds: logit(p) = base_value + sum(shap_values).")

            sh_c1, sh_c2 = st.columns(2)
            with sh_c1:
                st.markdown("#### 🚨 Top Risk-Increasing Drivers (Pushed Toward Fraud)")
                if local_exp["top_risk_increasing_factors"]:
                    up_df = pd.DataFrame([
                        {
                            "Feature": f["feature"],
                            "Raw Input Value": f["raw_value"],
                            "SHAP Impact (+ log-odds)": f"+{f['shap_value']:.4f}",
                        }
                        for f in local_exp["top_risk_increasing_factors"]
                    ])
                    st.table(up_df)
                else:
                    st.write("No features elevated the score above baseline.")

            with sh_c2:
                st.markdown("#### 🛡️ Top Risk-Mitigating Drivers (Pushed Toward Legitimacy)")
                if local_exp["top_risk_decreasing_factors"]:
                    down_df = pd.DataFrame([
                        {
                            "Feature": f["feature"],
                            "Raw Input Value": f["raw_value"],
                            "SHAP Impact (- log-odds)": f"{f['shap_value']:.4f}",
                        }
                        for f in local_exp["top_risk_decreasing_factors"]
                    ])
                    st.table(down_df)
                else:
                    st.write("No features significantly lowered the score.")

            st.info(
                f"**Attribution Sanity Check:**  \n"
                f"Base Prior Log-Odds: `{local_exp['base_value_log_odds']}` (Base Probability: `{local_exp['base_probability']:.4%}`)  \n"
                f"Total Log-Odds Shift: `{local_exp['total_log_odds']}` $\\rightarrow$ Final Posterior Fraud Probability: `{local_exp['fraud_probability']:.4%}`."
            )

    # Recent Predictions Section
    if st.session_state["prediction_history"]:
        st.markdown("---")
        st.markdown("### 🕒 Recent Transaction History (This Session)")
        history_df = pd.DataFrame(st.session_state["prediction_history"]).iloc[::-1]
        st.dataframe(history_df, hide_index=True)


# ==============================================================================
# TAB 4: Mathematical & Behavioral Theory
# ==============================================================================
elif tab_choice == "📐 Mathematical & Behavioral Theory":
    st.subheader("Mathematical Formulations & Behavioral Economics")

    st.markdown("""
    ### 1. Weighted Asymmetric Logistic Loss
    Standard cross-entropy penalizes false positives and false negatives equally:
    $$\\mathcal{L}_{\\text{symmetric}} = - [y \\ln \\sigma(p) + (1-y) \\ln (1 - \\sigma(p))]$$

    In fraud detection, **missing a fraud ($FN$) is significantly more expensive than an alert ($FP$)**.
    Under explicit cost parameters $C_{FN}$ and $C_{FP}$, the expected loss is:
    $$\\mathcal{L}_{\\text{asymmetric}} = - [C_{FN} \\cdot y \\ln \\sigma(p) + C_{FP} \\cdot (1-y) \\ln (1 - \\sigma(p))]$$
    """)

    st.markdown("""
    ### 2. Analytical Gradient & Hessian
    XGBoost fits additive regression trees using a second-order Taylor expansion requiring exact first and second derivatives:

    **Gradient ($g_i$):**
    $$g_i = \\frac{\\partial \\mathcal{L}}{\\partial p_i} = - C_{FN} \\cdot y_i (1 - \\sigma(p_i)) + C_{FP} \\cdot (1 - y_i) \\sigma(p_i)$$

    **Hessian ($h_i$):**
    $$h_i = \\frac{\\partial^2 \\mathcal{L}}{\\partial p_i^2} = \\sigma(p_i)(1 - \\sigma(p_i)) \\left[ C_{FN} \\cdot y_i + C_{FP} \\cdot (1 - y_i) \\right]$$

    Notice that since $\\sigma(p) \\in (0, 1)$ and $C_{FN}, C_{FP} > 0$, the Hessian is **strictly positive ($h_i > 0$) for all $p$**,
    guaranteeing strict convexity and stable Newton-Raphson tree weight updates.
    """)

    st.markdown("""
    ### 3. Bayes Risk-Minimizing Threshold
    By minimizing expected risk $\\mathbb{E}[\\text{Cost}] = (1-p) C_{FP} \\mathbb{I}_{\\hat{y}=1} + p C_{FN} \\mathbb{I}_{\\hat{y}=0}$,
    the theoretical optimal decision threshold is:
    $$\\tau^* = \\frac{C_{FP}}{C_{FP} + C_{FN}}$$
    With $C_{FP} = 5$ and $C_{FN} = 20$:
    $$\\tau^* = \\frac{5}{5 + 20} = 0.200$$
    This Bayes threshold is a reference for calibrated probabilities under the stated two-cost decision rule.
    """)

    if metrics and "asymmetric_model" in metrics:
        current_threshold = metrics["asymmetric_model"]["threshold"]
        st.markdown(
            f"The saved asymmetric model's validation-selected threshold is **{current_threshold:.3f}**. "
            "It is selected empirically on the validation split and can differ from the simple theoretical reference above."
        )

    st.markdown(r"""
    ### 4. Prospect Theory (Kahneman & Tversky, 1979)
    Human decision-makers perceive losses more acutely than equivalent gains:
    $$v(x) = \begin{cases} x^\alpha & \text{if } x \ge 0 \\ -\lambda (-x)^\beta & \text{if } x < 0 \end{cases}$$
    Where $\lambda \approx 2.25$ is the empirical loss aversion parameter.
    In financial crime risk management, fraud managers exhibit even higher loss aversion ($\lambda \approx 4\text{--}20\times$)
    due to chargeback penalties and reputational consequences.
    """)

    pt_img = os.path.join(ROOT, "outputs", "plots", "prospect_theory_curve.png")
    if os.path.exists(pt_img):
        st.image(pt_img, caption="Prospect Theory Value Function vs Linear Utility")
