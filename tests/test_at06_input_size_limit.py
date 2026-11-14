
"""
AT06 - Input Size Limit Testing

Kiem tra gioi han gia tri dau vao cua API:
1. Du lieu nam trong gioi han duoc chap nhan.
2. Gia tri vuot nguong duong bi tu choi.
3. Gia tri vuot nguong am bi tu choi.
4. Gia tri bang nguong toi da duoc chap nhan.
5. Gia tri vuot nguong rat lon bi tu choi.
"""

import numpy as np
import pytest

from fastapi.testclient import TestClient

from api import main


LIMIT = 1_000_000


@pytest.fixture
def client(monkeypatch):
    """Tao TestClient voi cau hinh gioi han co dinh."""

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv(
        "API_MAX_ABS_INPUT",
        str(LIMIT)
    )

    with main._rate_lock:
        main._request_times.clear()

    return TestClient(main.app)


def make_values(value=0.0):
    """Tao ma tran dau vao 96 x 7."""

    return [
        [float(value)] * 7
        for _ in range(96)
    ]


def mock_prediction(monkeypatch):
    """Mock inference de khong can tai model that."""

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


def test_at06_1_input_within_limit(client, monkeypatch):
    """Gia tri hop le trong gioi han."""

    mock_prediction(monkeypatch)

    response = client.post(
        "/forecast",
        json={
            "values": make_values(100.0)
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["forecast_pm25"]) == 24


def test_at06_2_positive_value_exceeds_limit(client):
    """Gia tri duong vuot nguong."""

    values = make_values(100.0)
    values[0][0] = LIMIT + 1

    response = client.post(
        "/forecast",
        json={"values": values}
    )

    assert response.status_code == 422


def test_at06_3_negative_value_exceeds_limit(client):
    """Gia tri am vuot nguong."""

    values = make_values(100.0)
    values[0][0] = -(LIMIT + 1)

    response = client.post(
        "/forecast",
        json={"values": values}
    )

    assert response.status_code == 422


def test_at06_4_value_exactly_at_limit(client, monkeypatch):
    """Gia tri bang nguong toi da."""

    mock_prediction(monkeypatch)

    response = client.post(
        "/forecast",
        json={
            "values": make_values(LIMIT)
        }
    )

    assert response.status_code == 200


def test_at06_5_extremely_large_value(client):
    """Gia tri cuc lon phai bi tu choi."""

    values = make_values(0.0)
    values[95][6] = 1e12

    response = client.post(
        "/forecast",
        json={"values": values}
    )

    assert response.status_code == 422
