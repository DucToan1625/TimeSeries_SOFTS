
import numpy as np


def test_tc01_forecast_with_valid_data(
    client,
    valid_payload,
    auth_headers
):
    """
    TC01 - Kiểm thử API dự báo với dữ liệu hợp lệ.

    Input:
        96 giờ dữ liệu
        7 biến môi trường

    Expected:
        HTTP 200
        24 giá trị dự báo PM2.5
        Không có NaN hoặc Infinity
    """

    response = client.post(
        "/forecast",
        json=valid_payload,
        headers=auth_headers
    )

    # Kiểm tra HTTP status
    assert response.status_code == 200, (
        f"TC01 thất bại. HTTP {response.status_code}: "
        f"{response.text}"
    )

    result = response.json()

    # Kiểm tra các trường trong response
    assert "forecast_pm25" in result
    assert "horizon_hours" in result
    assert "unit" in result
    assert "model" in result

    # Kiểm tra số lượng giá trị dự báo
    predictions = np.asarray(
        result["forecast_pm25"],
        dtype=np.float64
    )

    assert predictions.shape == (24,), (
        f"Yêu cầu 24 giá trị, nhận được "
        f"{predictions.shape}"
    )

    # Kiểm tra giá trị hữu hạn
    assert np.isfinite(predictions).all(), (
        "Kết quả chứa NaN hoặc Infinity"
    )

    # Kiểm tra thông tin mô hình
    assert result["horizon_hours"] == 24
    assert result["model"] == "SOFTS"
    assert result["unit"] == "ug/m3"

    print("\n========== TC01 RESULT ==========")
    print("HTTP Status:", response.status_code)
    print("Model:", result["model"])
    print("Forecast horizon:", result["horizon_hours"])
    print("Unit:", result["unit"])
    print("Number of predictions:", len(predictions))
    print("First 5 predictions:", predictions[:5])
    print("MAE/RMSE: Chưa tính trong TC01")
    print("================================")
