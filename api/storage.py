
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


# Đường dẫn lưu lịch sử dự báo.
# Có thể thay đổi bằng biến môi trường FORECAST_STORAGE_FILE.
ROOT = Path(__file__).resolve().parents[1]

STORAGE_FILE = Path(
    os.getenv(
        "FORECAST_STORAGE_FILE",
        str(ROOT / "data" / "forecast_history.csv")
    )
)

_storage_lock = Lock()


def save_forecast(forecast_pm25: list[float]) -> None:
    """
    Lưu kết quả dự báo PM2.5 vào CSV.

    Mỗi lần dự báo thành công sẽ thêm một bản ghi gồm:
    - timestamp: thời điểm dự báo UTC
    - model: tên mô hình
    - horizon_hours: số giờ dự báo
    - forecast_pm25: 24 giá trị dự báo
    """

    if len(forecast_pm25) != 24:
        raise ValueError("Forecast must contain exactly 24 values")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "SOFTS",
        "horizon_hours": 24,
        "forecast_pm25": json.dumps(forecast_pm25)
    }

    with _storage_lock:
        STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

        file_exists = STORAGE_FILE.exists()
        write_header = not file_exists or STORAGE_FILE.stat().st_size == 0

        with STORAGE_FILE.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "model",
                    "horizon_hours",
                    "forecast_pm25"
                ]
            )

            if write_header:
                writer.writeheader()

            writer.writerow(record)

