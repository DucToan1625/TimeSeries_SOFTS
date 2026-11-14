
"""
AT03 - Cross-Site Scripting (XSS) Security Test

Kiem tra:
1. Script tag injection.
2. HTML onerror injection.
3. SVG onload injection.
4. JavaScript URL injection.
5. API van hoat dong sau cac request XSS.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import requests


BASE_URL = os.getenv(
    "SOFTS_TEST_API_URL",
    "http://127.0.0.1:8000"
)

API_KEY = os.getenv(
    "SOFTS_TEST_API_KEY",
    "at02-test-token"
)

TIMEOUT = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "windows_96_to_24.npz"


def load_valid_input():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataset = np.load(DATA_FILE)
    values = dataset["X_test"][0].astype(float).tolist()

    assert len(values) == 96
    assert all(len(row) == 7 for row in values)

    return values


def send_xss_payload(payload):
    values = load_valid_input()

    # Chen chuoi XSS vao vi tri du lieu PM2.5 dau tien
    values[0][0] = payload

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    return requests.post(
        f"{BASE_URL}/forecast",
        json={"values": values},
        headers=headers,
        timeout=TIMEOUT
    )


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "javascript:alert('XSS')",
    ],
    ids=[
        "script_tag",
        "html_onerror",
        "svg_onload",
        "javascript_url",
    ]
)
def test_at03_xss_payload_rejected(payload):
    response = send_xss_payload(payload)

    assert response.status_code == 422, (
        "XSS payload was not rejected. "
        f"Expected HTTP 422, got {response.status_code}. "
        f"Response: {response.text}"
    )

    # API phai tra ve JSON, khong phai trang HTML
    content_type = response.headers.get(
        "content-type", ""
    ).lower()

    assert "application/json" in content_type, (
        f"Unexpected Content-Type: {content_type}"
    )


def test_at03_api_health_after_xss():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "SOFTS"
