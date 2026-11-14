
import json
import os

from fastapi.testclient import TestClient

from api.main import app


TEST_API_KEY = os.getenv(
    "API_KEY",
    "softs-test-secret-2026"
)

client = TestClient(
    app,
    headers={
        "Authorization": f"Bearer {TEST_API_KEY}"
    }
)


def create_valid_input():
    """Tao du lieu dau vao co kich thuoc 96x7."""
    return [[1.0 for _ in range(7)] for _ in range(96)]


def check_rejected(values):
    """
    Gui JSON tho de kiem tra API.
    allow_nan=True cho phep tao JSON co NaN va Infinity.
    """

    body = json.dumps(
        {"values": values},
        allow_nan=True
    )

    response = client.post(
        "/forecast",
        content=body,
        headers={
            "Content-Type": "application/json"
        }
    )

    assert response.status_code == 422, (
        f"Expected HTTP 422, received "
        f"{response.status_code}: {response.text}"
    )


def test_tc07_nan_value():
    """TC07.1: API tu choi gia tri NaN."""
    values = create_valid_input()
    values[10][0] = float("nan")

    check_rejected(values)


def test_tc07_positive_infinity():
    """TC07.2: API tu choi gia tri +Inf."""
    values = create_valid_input()
    values[20][0] = float("inf")

    check_rejected(values)


def test_tc07_negative_infinity():
    """TC07.3: API tu choi gia tri -Inf."""
    values = create_valid_input()
    values[30][0] = float("-inf")

    check_rejected(values)


def test_tc07_null_value():
    """TC07.4: API tu choi gia tri null."""
    values = create_valid_input()
    values[40][0] = None

    check_rejected(values)
