"""
Tests for Unified Decision Engine & Boundary Consistency
=========================================================
Verifies:
1. Complete mathematical monotonicity across threshold boundaries.
2. `is_fraud` strictly aligns with decision action (no ALLOW/CHALLENGE on is_fraud=True,
   no BLOCK/HARD_BLOCK on is_fraud=False).
3. Exact behavior at boundary points: tau - eps, tau, tau + eps.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from src.decision_engine import DecisionEngine


@pytest.fixture
def engine():
    # Use asymmetric model's threshold tau = 0.160
    return DecisionEngine(fraud_threshold=0.160, challenge_ratio=0.50, hard_block_ratio=0.60)


def test_threshold_ordering(engine):
    assert 0.0 < engine.threshold_challenge < engine.threshold < engine.threshold_hard_block < 1.0
    assert engine.threshold == 0.160
    assert engine.threshold_challenge == 0.080
    # 0.16 + (1 - 0.16) * 0.60 = 0.16 + 0.504 = 0.664
    assert engine.threshold_hard_block == 0.664


def test_boundary_low_to_medium(engine):
    eps = 1e-6
    # Just below challenge threshold
    d_below = engine.evaluate(engine.threshold_challenge - eps)
    assert d_below.risk_tier == "LOW"
    assert d_below.decision == "ALLOW"
    assert d_below.is_fraud is False

    # Exactly at challenge threshold
    d_at = engine.evaluate(engine.threshold_challenge)
    assert d_at.risk_tier == "MEDIUM"
    assert d_at.decision == "CHALLENGE"
    assert d_at.is_fraud is False

    # Just above challenge threshold
    d_above = engine.evaluate(engine.threshold_challenge + eps)
    assert d_above.risk_tier == "MEDIUM"
    assert d_above.decision == "CHALLENGE"
    assert d_above.is_fraud is False


def test_boundary_medium_to_high_fraud_boundary(engine):
    eps = 1e-6
    # Just below fraud threshold -> CHALLENGE, NOT fraud
    d_below = engine.evaluate(engine.threshold - eps)
    assert d_below.risk_tier == "MEDIUM"
    assert d_below.decision == "CHALLENGE"
    assert d_below.is_fraud is False

    # Exactly at fraud threshold -> BLOCK, IS fraud
    d_at = engine.evaluate(engine.threshold)
    assert d_at.risk_tier == "HIGH"
    assert d_at.decision == "BLOCK"
    assert d_at.is_fraud is True

    # Just above fraud threshold -> BLOCK, IS fraud
    d_above = engine.evaluate(engine.threshold + eps)
    assert d_above.risk_tier == "HIGH"
    assert d_above.decision == "BLOCK"
    assert d_above.is_fraud is True


def test_boundary_high_to_critical(engine):
    eps = 1e-6
    # Just below hard block threshold
    d_below = engine.evaluate(engine.threshold_hard_block - eps)
    assert d_below.risk_tier == "HIGH"
    assert d_below.decision == "BLOCK"
    assert d_below.is_fraud is True

    # Exactly at hard block threshold
    d_at = engine.evaluate(engine.threshold_hard_block)
    assert d_at.risk_tier == "CRITICAL"
    assert d_at.decision == "HARD_BLOCK"
    assert d_at.is_fraud is True

    # High end
    d_max = engine.evaluate(1.0)
    assert d_max.risk_tier == "CRITICAL"
    assert d_max.decision == "HARD_BLOCK"
    assert d_max.is_fraud is True


def test_decision_and_fraud_consistency_across_grid(engine):
    """Scan 1,000 probabilities from 0.0 to 1.0 and guarantee no contradiction."""
    for p in [i / 1000.0 for i in range(1001)]:
        res = engine.evaluate(p)
        if res.is_fraud:
            assert res.decision in ["BLOCK", "HARD_BLOCK"]
            assert res.risk_tier in ["HIGH", "CRITICAL"]
            assert p >= engine.threshold
        else:
            assert res.decision in ["ALLOW", "CHALLENGE"]
            assert res.risk_tier in ["LOW", "MEDIUM"]
            assert p < engine.threshold
