
"""
AT09 - CORS Security Testing

Kiem tra:
1. Origin khong tin cay khong duoc cap phep CORS.
2. API khong tra Access-Control-Allow-Origin: *.
3. Preflight request khong duoc tu dong cho phep.
4. Health endpoint khong cong khai CORS cho origin bat ky.

Luu y:
API hien tai chua khai bao CORSMiddleware.
"""

import pytest
from fastapi.testclient import TestClient

from api import main


UNTRUSTED_ORIGIN = "https://malicious-example.com"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def test_at09_1_untrusted_origin_not_allowed(client):
    """Origin khong tin cay khong duoc cap CORS."""

    response = client.get(
        "/health",
        headers={"Origin": UNTRUSTED_ORIGIN}
    )

    assert response.status_code == 200

    assert (
        "access-control-allow-origin"
        not in response.headers
    )


def test_at09_2_no_wildcard_cors(client):
    """API khong cho phep moi origin bang wildcard."""

    response = client.get(
        "/health",
        headers={"Origin": UNTRUSTED_ORIGIN}
    )

    allow_origin = response.headers.get(
        "access-control-allow-origin"
    )

    assert allow_origin != "*"


def test_at09_3_preflight_not_automatically_allowed(client):
    """Preflight request khong duoc tu dong cho phep."""

    response = client.options(
        "/forecast",
        headers={
            "Origin": UNTRUSTED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": (
                "authorization, content-type"
            )
        }
    )

    assert response.status_code == 405

    assert (
        "access-control-allow-origin"
        not in response.headers
    )


def test_at09_4_forecast_does_not_allow_untrusted_origin(
    client,
    monkeypatch
):
    """Forecast khong tra CORS header cho origin la."""

    monkeypatch.setattr(
        main,
        "predict",
        lambda values: [1.0] * 24
    )

    monkeypatch.setattr(
        main,
        "save_forecast",
        lambda values: None
    )

    response = client.post(
        "/forecast",
        json={
            "values": [
                [0.0] * 7
                for _ in range(96)
            ]
        },
        headers={
            "Origin": UNTRUSTED_ORIGIN
        }
    )

    assert response.status_code == 200

    assert (
        "access-control-allow-origin"
        not in response.headers
    )
