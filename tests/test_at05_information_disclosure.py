
"""
AT05 - Sensitive Information Disclosure Testing

Kiem tra API co lo lo thong tin noi bo khi:
1. Du lieu dau vao khong hop le.
2. Qua trinh inference gap loi.
3. Qua trinh luu CSV gap loi.
4. Truy cap endpoint health.
5. Request bi tu choi xac thuc.

Muc tieu:
Khong de lo traceback, duong dan noi bo,
thong tin checkpoint hoac du lieu nhay cam.
"""

import pytest

from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    """
    Tao TestClient va vo hieu hoa API Key
    de kiem thu doc lap voi cau hinh moi truong.
    """

    monkeypatch.delenv("API_KEY", raising=False)

    # Xoa rate limit giua cac lan kiem thu
    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def test_at05_1_validation_error_does_not_leak_input(client):
    """
    Du lieu dau vao sai khong duoc xuat hien
    trong noi dung phan hoi loi.
    """

    secret_marker = "CONFIDENTIAL_INPUT_987654"

    response = client.post(
        "/forecast",
        json={
            "values": [
                [secret_marker] + [0.0] * 6
            ]
        }
    )

    assert response.status_code == 422

    body = response.text.lower()

    assert secret_marker.lower() not in body
    assert "traceback" not in body
    assert "site-packages" not in body


def test_at05_2_inference_error_is_sanitized(
    client,
    monkeypatch
):
    """
    Loi inference khong duoc tiet lo traceback
    hoac duong dan checkpoint.
    """

    secret_path = (
        "C:\\PRIVATE_SERVER\\checkpoint_secret.pth"
    )

    def fake_predict(values):
        raise RuntimeError(
            f"Internal error at {secret_path}"
        )

    monkeypatch.setattr(
        main,
        "predict",
        fake_predict
    )

    response = client.post(
        "/forecast",
        json={
            "values": [
                [0.0] * 7
                for _ in range(96)
            ]
        }
    )

    assert response.status_code == 500

    body = response.text.lower()

    assert "forecast inference failed" in body
    assert secret_path.lower() not in body
    assert "traceback" not in body
    assert "runtimeerror" not in body


def test_at05_3_storage_error_is_sanitized(
    client,
    monkeypatch
):
    """
    Loi luu CSV khong duoc tiet lo duong dan
    hoac thong tin he thong.
    """

    secret_path = (
        "C:\\PRIVATE_SERVER\\forecast_history.csv"
    )

    monkeypatch.setattr(
        main,
        "predict",
        lambda values: [1.0] * 24
    )

    def fake_save_forecast(values):
        raise OSError(
            f"Cannot access {secret_path}"
        )

    monkeypatch.setattr(
        main,
        "save_forecast",
        fake_save_forecast
    )

    response = client.post(
        "/forecast",
        json={
            "values": [
                [0.0] * 7
                for _ in range(96)
            ]
        }
    )

    assert response.status_code == 503

    body = response.text.lower()

    assert "forecast storage unavailable" in body
    assert secret_path.lower() not in body
    assert "traceback" not in body
    assert "oserror" not in body


def test_at05_4_health_does_not_expose_internal_paths(client):
    """
    Health endpoint khong duoc tra ve duong dan
    checkpoint, scaler hoac thu muc may chu.
    """

    response = client.get("/health")

    assert response.status_code == 200

    body = response.text.lower()

    assert "artifacts_present" in body
    assert "checkpoint.pth" not in body
    assert "scaler_stats.csv" not in body
    assert "c:\\users\\" not in body
    assert "traceback" not in body


def test_at05_5_authentication_error_is_sanitized(
    client,
    monkeypatch
):
    """
    Loi xac thuc khong duoc tiet lo API Key.
    """

    secret_key = "CONFIDENTIAL_API_KEY_12345"

    monkeypatch.setenv(
        "API_KEY",
        secret_key
    )

    response = client.post(
        "/forecast",
        json={
            "values": [
                [0.0] * 7
                for _ in range(96)
            ]
        }
    )

    assert response.status_code == 401

    body = response.text.lower()

    assert "invalid or missing api token" in body
    assert secret_key.lower() not in body
    assert "traceback" not in body
