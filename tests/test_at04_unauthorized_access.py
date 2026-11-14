
"""
AT04 - Unauthorized Access Testing

Muc tieu:
1. Tu choi truy cap khi khong co Authorization header.
2. Tu choi truy cap khi Bearer token rong.
3. Tu choi truy cap khi token khong hop le.
4. Tu choi truy cap khi sai dinh dang Authorization.
5. Xac nhan endpoint health van hoat dong cong khai.

Luu y:
- API_KEY phai duoc cau hinh tren tien trinh Uvicorn.
- Test khong tu dong thay doi cau hinh server.
"""

import os

import numpy as np
import requests


BASE_URL = os.getenv(
    "SOFTS_TEST_API_URL",
    "http://127.0.0.1:8000"
)

TIMEOUT = 60

API_KEY = os.getenv(
    "SOFTS_TEST_API_KEY",
    "at02-test-token"
)


def load_valid_input():
    """Load input hop le tu bo du lieu kiem thu."""

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    data_file = os.path.join(
        project_root,
        "data",
        "windows_96_to_24.npz"
    )

    if not os.path.isfile(data_file):
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    dataset = np.load(data_file)

    values = dataset["X_test"][0].astype(
        float
    ).tolist()

    assert len(values) == 96
    assert all(len(row) == 7 for row in values)

    return values


def send_forecast(headers=None):
    """Gui request den endpoint forecast."""

    return requests.post(
        f"{BASE_URL}/forecast",
        json={
            "values": load_valid_input()
        },
        headers=headers or {},
        timeout=TIMEOUT
    )


def test_at04_1_access_without_authorization():
    """Khong co Authorization header."""

    response = send_forecast()

    assert response.status_code == 401, (
        "Expected HTTP 401 without Authorization, "
        f"got {response.status_code}: {response.text}"
    )


def test_at04_2_access_with_empty_bearer_token():
    """Bearer token rong."""

    response = send_forecast({
        "Authorization": "Bearer "
    })

    assert response.status_code == 401, (
        "Expected HTTP 401 with empty Bearer token, "
        f"got {response.status_code}: {response.text}"
    )


def test_at04_3_access_with_invalid_token():
    """Token khong hop le."""

    response = send_forecast({
        "Authorization": "Bearer unauthorized-token"
    })

    assert response.status_code == 401, (
        "Expected HTTP 401 with invalid token, "
        f"got {response.status_code}: {response.text}"
    )


def test_at04_4_access_with_wrong_authorization_scheme():
    """Sai dinh dang xac thuc."""

    response = send_forecast({
        "Authorization": f"Basic {API_KEY}"
    })

    assert response.status_code == 401, (
        "Expected HTTP 401 with wrong authorization scheme, "
        f"got {response.status_code}: {response.text}"
    )


def test_at04_5_public_health_endpoint():
    """Health endpoint duoc phep truy cap cong khai."""

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT
    )

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}"
    )

    data = response.json()

    assert data["model"] == "SOFTS"
    assert data["status"] in (
        "ready",
        "not_loaded"
    )
