
"""
AT02 - API Abuse / Brute Force Security Test

Kiem tra:
1. Khong co API Key -> HTTP 401.
2. API Key sai -> HTTP 401.
3. Vuot gioi han request -> HTTP 429.
4. API van hoat dong sau khi bi gioi han.
"""

import os
from pathlib import Path

import numpy as np
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "SOFTS_TEST_API_URL",
    "http://127.0.0.1:8000"
)

API_KEY = os.getenv(
    "SOFTS_TEST_API_KEY",
    "at02-test-token"
)

RATE_LIMIT = 5

TIMEOUT = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "windows_96_to_24.npz"


# ============================================================
# 2. LOAD VALID INPUT
# ============================================================

def load_valid_input():
    """Load one valid sample with shape (96, 7)."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataset = np.load(DATA_FILE)

    values = dataset["X_test"][0].astype(float).tolist()

    assert len(values) == 96
    assert all(len(row) == 7 for row in values)

    return values


# ============================================================
# 3. HTTP REQUEST HELPERS
# ============================================================

def send_request(token=None):
    """Send a forecast request with optional Bearer token."""

    headers = {}

    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    return requests.post(
        f"{BASE_URL}/forecast",
        json={"values": load_valid_input()},
        headers=headers,
        timeout=TIMEOUT
    )


# ============================================================
# AT02.1 - REQUEST WITHOUT API KEY
# ============================================================

def test_at02_1_request_without_api_key():
    """Requests without authentication must be rejected."""

    response = send_request()

    assert response.status_code == 401, (
        f"Expected HTTP 401, got {response.status_code}. "
        "Check whether API_KEY is configured on the server."
    )


# ============================================================
# AT02.2 - REQUEST WITH INVALID API KEY
# ============================================================

def test_at02_2_request_with_invalid_api_key():
    """Requests with an invalid token must be rejected."""

    response = send_request("invalid-token")

    assert response.status_code == 401, (
        f"Expected HTTP 401, got {response.status_code}"
    )


# ============================================================
# AT02.3 - RATE LIMITING
# ============================================================

def test_at02_3_rate_limiting():
    """
    With API_RATE_LIMIT_PER_MINUTE=5:
    - First five requests should be accepted.
    - Sixth request should return HTTP 429.
    """

    accepted = 0

    for index in range(RATE_LIMIT):

        response = send_request(API_KEY)

        assert response.status_code == 200, (
            f"Request {index + 1} failed. "
            f"Expected HTTP 200, got {response.status_code}: "
            f"{response.text}"
        )

        accepted += 1

        data = response.json()

        assert len(data["forecast_pm25"]) == 24

    assert accepted == RATE_LIMIT

    # Sixth request must exceed the configured limit.
    blocked_response = send_request(API_KEY)

    assert blocked_response.status_code == 429, (
        f"Expected HTTP 429 after {RATE_LIMIT} requests, "
        f"got {blocked_response.status_code}: "
        f"{blocked_response.text}"
    )


# ============================================================
# AT02.4 - API HEALTH AFTER RATE LIMIT
# ============================================================

def test_at02_4_api_health_after_rate_limit():
    """Health endpoint should remain accessible."""

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "SOFTS"
