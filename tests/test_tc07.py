
import json
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def create_valid_input():
    """Tạo dữ liệu đầu vào có kích thước 96x7."""
    return [[1.0 for _ in range(7)] for _ in range(96)]


def check_rejected(values):
    """
    Gửi JSON thô để kiểm tra API.
    allow_nan=True cho phép tạo JSON có NaN và Infinity
    phục vụ kiểm thử validator của API.
    """
    body = json.dumps(
        {"values": values},
        allow_nan=True
    )

    response = client.post(
        "/forecast",
        content=body,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422, (
        f"Expected HTTP 422, received "
        f"{response.status_code}: {response.text}"
    )


def test_tc07_nan_value():
    """TC07.1: API từ chối giá trị NaN."""
    values = create_valid_input()
    values[10][0] = float("nan")

    check_rejected(values)


def test_tc07_positive_infinity():
    """TC07.2: API từ chối giá trị +Inf."""
    values = create_valid_input()
    values[20][0] = float("inf")

    check_rejected(values)


def test_tc07_negative_infinity():
    """TC07.3: API từ chối giá trị -Inf."""
    values = create_valid_input()
    values[30][0] = float("-inf")

    check_rejected(values)


def test_tc07_null_value():
    """TC07.4: API từ chối giá trị null."""
    values = create_valid_input()
    values[40][0] = None

    check_rejected(values)
