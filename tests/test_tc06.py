
import os
import numpy as np
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

def create_normal_input():
    """Tạo dữ liệu đầu vào 96 giờ, gồm 7 biến."""
    rng = np.random.default_rng(42)

    values = np.zeros((96, 7), dtype=np.float32)

    values[:, 0] = rng.uniform(20, 80, 96)       # PM2.5
    values[:, 1] = rng.uniform(30, 120, 96)      # PM10
    values[:, 2] = rng.uniform(20, 35, 96)       # Temperature
    values[:, 3] = rng.uniform(40, 90, 96)       # Humidity
    values[:, 4] = rng.uniform(0, 10, 96)        # Wind speed
    values[:, 5] = rng.uniform(990, 1020, 96)    # Pressure
    values[:, 6] = rng.uniform(0, 10, 96)        # Rainfall

    return values


def check_forecast_response(response):
    """Kiểm tra cấu trúc và tính hợp lệ của kết quả dự báo."""
    assert response.status_code == 200, (
        f"HTTP {response.status_code}: {response.text}"
    )

    result = response.json()

    assert "forecast_pm25" in result
    assert result["horizon_hours"] == 24
    assert result["model"] == "SOFTS"
    assert result["unit"] == "ug/m3"

    predictions = np.asarray(result["forecast_pm25"], dtype=float)

    assert predictions.shape == (24,)
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= 0)


def test_tc06_abrupt_pm25_increase():
    """TC06.1: Kiểm tra khi PM2.5 tăng đột ngột."""
    values = create_normal_input()

    # Tăng PM2.5 lên 500 µg/m³ trong 6 giờ cuối.
    values[-6:, 0] = 500

    response = client.post(
        "/forecast",
        json={"values": values.tolist()}
    )

    check_forecast_response(response)


def test_tc06_abrupt_pm25_decrease():
    """TC06.2: Kiểm tra khi PM2.5 giảm đột ngột."""
    values = create_normal_input()

    # Giảm PM2.5 xuống 0 trong 6 giờ cuối.
    values[-6:, 0] = 0

    response = client.post(
        "/forecast",
        json={"values": values.tolist()}
    )

    check_forecast_response(response)
