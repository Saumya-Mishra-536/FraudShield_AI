import httpx
import pytest

from src.dashboard_api import DashboardAPIError, request_risk_score


PAYLOAD = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 120.5, "Time": 36000.0}
RESPONSE = {
    "fraud_probability": 0.03,
    "probability_calibrated": True,
    "decision": "CHALLENGE",
    "risk_tier": "MEDIUM",
    "threshold_used": 0.0492,
    "explanation": ["V14 increased model risk."],
    "top_risk_increasing_factors": [
        {"feature": "V14", "raw_value": -4.5, "shap_value": 1.2}
    ],
    "top_risk_decreasing_factors": [],
    "raw_margin_attribution": {
        "base_value_log_odds": -3.0,
        "total_log_odds": -1.8,
        "raw_model_score": 0.1419,
    },
    "recommendation": "Request step-up verification.",
    "model_version": "asymmetric-v1.2",
}


def test_request_risk_score_posts_payload_and_api_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["payload"] = request.read()
        seen["api_key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json=RESPONSE)

    result, latency_ms = request_risk_score(
        "http://fraudshield.test/",
        "dashboard-secret",
        PAYLOAD,
        transport=httpx.MockTransport(handler),
    )

    assert seen["url"] == "http://fraudshield.test/risk-score"
    assert seen["api_key"] == "dashboard-secret"
    assert b'"Amount":120.5' in seen["payload"]
    assert result["decision"] == "CHALLENGE"
    assert result["raw_margin_attribution"]["raw_model_score"] == pytest.approx(0.1419)
    assert latency_ms >= 0


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "rejected the API key"),
        (422, "rejected the transaction fields"),
        (429, "rate limit was reached"),
        (503, "not ready"),
    ],
)
def test_request_risk_score_maps_api_errors_without_exposing_body(status_code, message):
    def handler(request):
        return httpx.Response(status_code, json={"detail": "private transaction-related error"})

    with pytest.raises(DashboardAPIError, match=message) as error:
        request_risk_score(
            "http://fraudshield.test",
            "",
            PAYLOAD,
            transport=httpx.MockTransport(handler),
        )

    assert "private transaction-related error" not in str(error.value)


def test_request_risk_score_rejects_malformed_response():
    def handler(request):
        return httpx.Response(200, json={"fraud_probability": 0.2})

    with pytest.raises(DashboardAPIError, match="incomplete scoring response"):
        request_risk_score(
            "http://fraudshield.test",
            "",
            PAYLOAD,
            transport=httpx.MockTransport(handler),
        )


def test_request_risk_score_rejects_invalid_api_url():
    with pytest.raises(DashboardAPIError, match="HTTP\\(S\\) base URL"):
        request_risk_score("https://user:secret@example.test", "", PAYLOAD)


def test_request_risk_score_reports_timeout_safely():
    def handler(request):
        raise httpx.ReadTimeout("private request context", request=request)

    with pytest.raises(DashboardAPIError, match="timed out") as error:
        request_risk_score(
            "http://fraudshield.test",
            "",
            PAYLOAD,
            transport=httpx.MockTransport(handler),
        )

    assert "private request context" not in str(error.value)
