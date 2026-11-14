
"""
AT10 - API Documentation Security Testing

Kiem tra:
1. Swagger UI co hoat dong.
2. ReDoc co hoat dong.
3. OpenAPI schema co the truy cap.
4. Schema khong tiet lo duong dan checkpoint noi bo.
5. Schema khong tiet lo API Key that.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def test_at10_1_swagger_documentation(client):
    """Kiem tra Swagger UI."""

    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_at10_2_redoc_documentation(client):
    """Kiem tra ReDoc."""

    response = client.get("/redoc")

    assert response.status_code == 200
    assert "redoc" in response.text.lower()


def test_at10_3_openapi_schema_available(client):
    """Kiem tra OpenAPI schema."""

    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "openapi" in schema
    assert "paths" in schema
    assert "/forecast" in schema["paths"]
    assert "/health" in schema["paths"]


def test_at10_4_schema_does_not_expose_checkpoint_path(client):
    """Schema khong duoc lo duong dan checkpoint."""

    response = client.get("/openapi.json")

    assert response.status_code == 200

    body = response.text.lower()

    assert "checkpoint.pth" not in body
    assert "scaler_stats.csv" not in body
    assert "c:\\users\\" not in body


def test_at10_5_schema_does_not_expose_api_key(client):
    """Schema khong duoc lo API Key."""

    secret_key = "AT10_SECRET_KEY_987654"

    response = client.get("/openapi.json")

    assert response.status_code == 200

    assert secret_key not in response.text
