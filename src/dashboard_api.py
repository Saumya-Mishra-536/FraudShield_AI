"""Small, safe HTTP client used by the Streamlit live-scoring view."""
from __future__ import annotations

import math
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx


class DashboardAPIError(RuntimeError):
    """A user-safe FastAPI connection, response, or validation error."""


def request_risk_score(
    api_url: str,
    api_key: str,
    transaction: dict[str, float],
    *,
    transport: httpx.BaseTransport | None = None,
) -> tuple[dict[str, Any], float]:
    """Call the detailed scoring endpoint and return its payload and round-trip ms.

    Error messages deliberately omit response bodies and request data so transaction
    values and server internals are not surfaced in the dashboard.
    """
    base_url = api_url.strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        parsed.port
    except ValueError as exc:
        raise DashboardAPIError(
            "FRAUDSHIELD_API_URL must be a valid HTTP(S) base URL."
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise DashboardAPIError(
            "FRAUDSHIELD_API_URL must be an HTTP(S) base URL without credentials or query parameters."
        )

    headers = {"X-API-Key": api_key} if api_key else {}
    started = perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0), transport=transport) as client:
            response = client.post(
                f"{base_url}/risk-score",
                json=transaction,
                headers=headers,
            )
    except httpx.TimeoutException as exc:
        raise DashboardAPIError(
            "The FastAPI request timed out. Check API readiness and try again."
        ) from exc
    except httpx.ConnectError as exc:
        raise DashboardAPIError(
            "The dashboard could not connect to FastAPI. Check FRAUDSHIELD_API_URL and API readiness."
        ) from exc
    except httpx.RequestError as exc:
        raise DashboardAPIError("The request to FastAPI failed. Check the service and try again.") from exc

    if response.status_code == 401:
        raise DashboardAPIError(
            "FastAPI rejected the API key. Set the same FRAUDSHIELD_API_KEY for the dashboard and API."
        )
    if response.status_code == 422:
        raise DashboardAPIError(
            "FastAPI rejected the transaction fields. Check that V1–V28, Amount, and Time are valid."
        )
    if response.status_code == 429:
        raise DashboardAPIError("The FastAPI rate limit was reached. Wait briefly and try again.")
    if response.status_code == 503:
        raise DashboardAPIError(
            "FastAPI is not ready. Check /health/ready and confirm model artifacts are available."
        )
    if not response.is_success:
        if response.status_code >= 500:
            raise DashboardAPIError(
                f"FastAPI returned an internal error (HTTP {response.status_code})."
            )
        raise DashboardAPIError(
            f"FastAPI rejected the request (HTTP {response.status_code})."
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise DashboardAPIError("FastAPI returned an unreadable scoring response.") from exc
    if not isinstance(result, dict):
        raise DashboardAPIError("FastAPI returned an invalid scoring response.")

    required = {
        "fraud_probability",
        "probability_calibrated",
        "decision",
        "risk_tier",
        "threshold_used",
        "explanation",
        "top_risk_increasing_factors",
        "top_risk_decreasing_factors",
        "raw_margin_attribution",
        "recommendation",
        "model_version",
    }
    missing = required.difference(result)
    if missing:
        raise DashboardAPIError("FastAPI returned an incomplete scoring response.")

    for field in ("fraud_probability", "threshold_used"):
        value = result[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise DashboardAPIError("FastAPI returned an invalid probability or threshold.")
    if not isinstance(result["probability_calibrated"], bool):
        raise DashboardAPIError("FastAPI returned an invalid calibration status.")
    for field in ("decision", "risk_tier", "recommendation", "model_version"):
        if not isinstance(result[field], str):
            raise DashboardAPIError("FastAPI returned an invalid scoring label.")
    if not isinstance(result["explanation"], list) or not all(
        isinstance(item, str) for item in result["explanation"]
    ):
        raise DashboardAPIError("FastAPI returned an invalid explanation.")
    for field in ("top_risk_increasing_factors", "top_risk_decreasing_factors"):
        factors = result[field]
        if not isinstance(factors, list):
            raise DashboardAPIError("FastAPI returned invalid SHAP attributions.")
        for factor in factors:
            if not isinstance(factor, dict) or not isinstance(factor.get("feature"), str):
                raise DashboardAPIError("FastAPI returned invalid SHAP attributions.")
            value = factor.get("shap_value")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise DashboardAPIError("FastAPI returned invalid SHAP attributions.")
            raw_value = factor.get("raw_value")
            if raw_value is not None and (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise DashboardAPIError("FastAPI returned invalid SHAP attributions.")

    attribution = result["raw_margin_attribution"]
    if not isinstance(attribution, dict):
        raise DashboardAPIError("FastAPI returned invalid raw-margin attributions.")
    for field in ("base_value_log_odds", "total_log_odds", "raw_model_score"):
        value = attribution.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise DashboardAPIError("FastAPI returned invalid raw-margin attributions.")

    return result, round((perf_counter() - started) * 1000.0, 1)
