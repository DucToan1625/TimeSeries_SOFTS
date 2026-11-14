
"""
AT08 - HTTP Method and Endpoint Security Testing

Kiem tra:
1. GET /forecast khong duoc ho tro.
2. PUT /forecast khong duoc ho tro.
3. DELETE /forecast khong duoc ho tro.
4. PATCH /forecast khong duoc ho tro.
5. Truy cap endpoint khong ton tai.
6. POST endpoint health khong duoc ho tro.
"""

import pytest

from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    """Tao TestClient doc lap."""

    monkeypatch.delenv("API_KEY", raising=False)

    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def test_at08_1_get_forecast_not_allowed(client):
    response = client.get("/forecast")

    assert response.status_code == 405


def test_at08_2_put_forecast_not_allowed(client):
    response = client.put(
        "/forecast",
        json={"values": []}
    )

    assert response.status_code == 405


def test_at08_3_delete_forecast_not_allowed(client):
    response = client.delete("/forecast")

    assert response.status_code == 405


def test_at08_4_patch_forecast_not_allowed(client):
    response = client.patch(
        "/forecast",
        json={"values": []}
    )

    assert response.status_code == 405


def test_at08_5_unknown_endpoint_not_found(client):
    response = client.get(
        "/internal-admin-secret"
    )

    assert response.status_code == 404


def test_at08_6_post_health_not_allowed(client):
    response = client.post(
        "/health",
        json={}
    )

    assert response.status_code == 405
