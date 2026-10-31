"""
FraudShield AI — Unified Decision and Risk Engine
=================================================
Centralized single source of truth for operational decision thresholds,
risk tier categorization, and action recommendations.

Guarantees:
1. `is_fraud` and `decision` never contradict each other:
   - If proba < threshold -> is_fraud = False, decision in {ALLOW, CHALLENGE}
   - If proba >= threshold -> is_fraud = True, decision in {BLOCK, HARD_BLOCK}
2. Predict endpoint, risk-score endpoint, and dashboard share identical logic.
3. Every threshold boundary is strictly monotonic and tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RiskDecision:
    fraud_probability: float
    is_fraud: bool
    risk_tier: str       # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    decision: str        # "ALLOW", "CHALLENGE", "BLOCK", "HARD_BLOCK"
    threshold_used: float
    threshold_challenge: float
    threshold_hard_block: float
    recommendation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "fraud_probability": round(self.fraud_probability, 6),
            "is_fraud": self.is_fraud,
            "risk_tier": self.risk_tier,
            "decision": self.decision,
            "threshold_used": round(self.threshold_used, 4),
            "threshold_challenge": round(self.threshold_challenge, 4),
            "threshold_hard_block": round(self.threshold_hard_block, 4),
            "recommendation": self.recommendation,
        }


class DecisionEngine:
    """
    Evaluates fraud probability against calibrated decision boundaries.
    """

    RECOMMENDATIONS: Dict[str, str] = {
        "ALLOW": "Process transaction normally without friction.",
        "CHALLENGE": "Step-up authentication required (SMS OTP or biometric confirmation).",
        "BLOCK": "Decline transaction immediately and notify cardholder for confirmation.",
        "HARD_BLOCK": "Critical fraud risk. Hard decline, freeze card, and route to financial crimes team.",
    }

    def __init__(
        self,
        fraud_threshold: float = 0.160,
        challenge_ratio: float = 0.50,
        hard_block_ratio: float = 0.60,
    ):
        """
        Args:
            fraud_threshold: Primary decision threshold tau (from model bundle or tuning).
            challenge_ratio: Ratio to establish tau_challenge = challenge_ratio * tau.
            hard_block_ratio: Ratio to establish tau_hard_block = tau + (1 - tau) * hard_block_ratio.
        """
        if not 0.0 < fraud_threshold < 1.0:
            raise ValueError(f"fraud_threshold must be in (0, 1), got {fraud_threshold}")

        self.threshold = float(fraud_threshold)
        self.threshold_challenge = round(float(self.threshold * challenge_ratio), 4)
        self.threshold_hard_block = round(
            float(self.threshold + (1.0 - self.threshold) * hard_block_ratio), 4
        )

        # Sanity assertion
        assert (
            0.0 < self.threshold_challenge < self.threshold < self.threshold_hard_block < 1.0
        ), "Threshold boundaries must be strictly increasing: 0 < tau_challenge < tau < tau_hard < 1"

    def evaluate(self, proba: float) -> RiskDecision:
        """
        Evaluate a single fraud probability into a consistent risk decision.

        Boundary rules:
        - [0.0, tau_challenge)        -> LOW       -> ALLOW        (is_fraud = False)
        - [tau_challenge, tau)        -> MEDIUM    -> CHALLENGE    (is_fraud = False)
        - [tau, tau_hard_block)       -> HIGH      -> BLOCK        (is_fraud = True)
        - [tau_hard_block, 1.0]       -> CRITICAL  -> HARD_BLOCK   (is_fraud = True)
        """
        p = float(proba)
        # Ensure within [0, 1]
        p = max(0.0, min(1.0, p))

        is_fraud = bool(p >= self.threshold)

        if p < self.threshold_challenge:
            tier = "LOW"
            action = "ALLOW"
        elif p < self.threshold:
            tier = "MEDIUM"
            action = "CHALLENGE"
        elif p < self.threshold_hard_block:
            tier = "HIGH"
            action = "BLOCK"
        else:
            tier = "CRITICAL"
            action = "HARD_BLOCK"

        return RiskDecision(
            fraud_probability=p,
            is_fraud=is_fraud,
            risk_tier=tier,
            decision=action,
            threshold_used=self.threshold,
            threshold_challenge=self.threshold_challenge,
            threshold_hard_block=self.threshold_hard_block,
            recommendation=self.RECOMMENDATIONS[action],
        )
