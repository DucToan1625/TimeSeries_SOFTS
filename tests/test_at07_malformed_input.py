
"""
AT07 - Malformed Input Testing

Kiem tra:
1. JSON sai dinh dang.
2. Thieu truong values.
3. Thua truong khong duoc khai bao.
4. values khong phai danh sach.
5. Du lieu khong dung cau truc 96 x 7.
6. Phan tu khong phai so.
"""

import pytest

from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    """Tao TestClient doc lap voi API Key."""

    monkeypatch.delenv("API_KEY", raising=False)

    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def valid_values():
    """Tao du lieu hop le 96 x 7."""

    return [
        [0.0] * 7
        for _ in range(96)
    ]


def test_at07_1_invalid_json_format(client):
    """JSON sai cu phap phai bi tu choi."""

    response = client.post(
        "/forecast",
        content='{"values": [invalid json}',
        headers={
            "Content-Type": "application/json"
        }
    )

    assert response.status_code == 422


def test_at07_2_missing_values_field(client):
    """Thieu truong values."""

    response = client.post(
        "/forecast",
        json={}
    )

    assert response.status_code == 422


def test_at07_3_extra_unknown_field(client):
    """Truong du khong duoc khai bao."""

    response = client.post(
        "/forecast",
        json={
            "values": valid_values(),
            "admin": True
        }
    )

    assert response.status_code == 422


def test_at07_4_values_is_not_a_list(client):
    """values khong phai danh sach."""

    response = client.post(
        "/forecast",
        json={
            "values": "invalid"
        }
    )

    assert response.status_code == 422


def test_at07_5_wrong_matrix_dimensions(client):
    """Ma tran khong dung kich thuoc 96 x 7."""

    response = client.post(
        "/forecast",
        json={
            "values": [
                [0.0] * 7
                for _ in range(95)
            ]
        }
    )

    assert response.status_code == 422


def test_at07_6_invalid_element_type(client):
    """Phan tu chuoi khong the chuyen thanh so."""

    values = valid_values()
    values[0][0] = "invalid-number"

    response = client.post(
        "/forecast",
        json={
            "values": values
        }
    )

    assert response.status_code == 422
