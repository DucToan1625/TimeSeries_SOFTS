
import csv
import json

from fastapi.testclient import TestClient

import api.main as main
import api.storage as storage


client = TestClient(main.app)


def create_valid_input():
    """Tạo dữ liệu đầu vào có kích thước 96 x 7."""
    return [
        [
            35.0,
            50.0,
            28.0,
            70.0,
            2.0,
            1005.0,
            0.0
        ]
        for _ in range(96)
    ]


def test_tc09_1_save_forecast_to_csv(
    tmp_path,
    monkeypatch
):
    """TC09.1: Kiểm tra lưu kết quả dự báo vào CSV."""

    csv_file = tmp_path / "forecast_history.csv"

    monkeypatch.setattr(
        storage,
        "STORAGE_FILE",
        csv_file
    )

    forecast_values = [
        float(i)
        for i in range(24)
    ]

    storage.save_forecast(forecast_values)

    assert csv_file.exists()

    with csv_file.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    record = rows[0]

    assert record["model"] == "SOFTS"
    assert int(record["horizon_hours"]) == 24

    saved_values = json.loads(
        record["forecast_pm25"]
    )

    assert len(saved_values) == 24
    assert saved_values == forecast_values

    print("\nTC09.1: CSV storage successful")
    print("Records:", len(rows))
    print("Forecast values:", len(saved_values))


def test_tc09_2_storage_failure_returns_503(
    monkeypatch
):
    """TC09.2: Kiểm tra API khi hệ thống lưu trữ gặp lỗi."""

    fake_forecast = [
        20.0 + i
        for i in range(24)
    ]

    # Giả lập mô hình dự báo thành công
    monkeypatch.setattr(
        main,
        "predict",
        lambda values: fake_forecast
    )

    # Giả lập lỗi ghi dữ liệu
    def mock_storage_failure(values):
        raise OSError("Simulated storage failure")

    monkeypatch.setattr(
        main,
        "save_forecast",
        mock_storage_failure
    )

    response = client.post(
        "/forecast",
        json={
            "values": create_valid_input()
        }
    )

    assert response.status_code == 503

    assert response.json()["detail"] == (
        "Forecast storage unavailable"
    )

    assert "Simulated storage failure" not in response.text

    print("\nTC09.2: Storage failure handled")
    print("HTTP status:", response.status_code)
    print("Response:", response.json())
