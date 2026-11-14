
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. KHAI BÁO ĐƯỜNG DẪN VÀ THÔNG SỐ
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

NPZ_PATH = BASE_DIR / "data" / "windows_96_to_24.npz"
PREDICTION_PATH = BASE_DIR / "forecast_api.json"

OUTPUT_CSV = BASE_DIR / "api_forecast_comparison.csv"
OUTPUT_PLOT = BASE_DIR / "api_forecast_comparison.png"
OUTPUT_METRICS = BASE_DIR / "api_forecast_metrics.txt"

PM25_MEAN = 49.737933
PM25_SCALE = 31.328638

# ==========================================
# 2. ĐỌC KẾT QUẢ DỰ BÁO TỪ REST API
# ==========================================

if not PREDICTION_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file: {PREDICTION_PATH}"
    )

with open(PREDICTION_PATH, "r", encoding="utf-8-sig") as f:
    prediction_data = json.load(f)

# Hỗ trợ cả JSON dạng danh sách và JSON dạng object
if isinstance(prediction_data, dict):
    forecast = prediction_data["forecast_pm25"]
else:
    forecast = prediction_data

forecast = np.asarray(forecast, dtype=np.float64).reshape(-1)

# ==========================================
# 3. ĐỌC DỮ LIỆU THỰC TẾ
# ==========================================

if not NPZ_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file: {NPZ_PATH}"
    )

data = np.load(NPZ_PATH)

# Lấy mẫu đầu tiên của tập kiểm thử
y_test = data["Y_test"][0]

# Chuyển về vector 24 giá trị
y_test = np.asarray(y_test).reshape(-1)

# Đưa dữ liệu thực tế về đơn vị µg/m³
actual = y_test * PM25_SCALE + PM25_MEAN

# ==========================================
# 4. KIỂM TRA DỮ LIỆU
# ==========================================

if len(forecast) != 24:
    raise ValueError(
        f"API phải trả về 24 giá trị, hiện có {len(forecast)}"
    )

if len(actual) != 24:
    raise ValueError(
        f"Dữ liệu thực tế phải có 24 giá trị, hiện có {len(actual)}"
    )

if not np.isfinite(forecast).all():
    raise ValueError("Dự báo API chứa NaN hoặc Inf")

if not np.isfinite(actual).all():
    raise ValueError("Dữ liệu thực tế chứa NaN hoặc Inf")

# ==========================================
# 5. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ==========================================

errors = forecast - actual

mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)

# MAPE chỉ tính khi giá trị thực tế khác 0
nonzero_mask = np.abs(actual) > 1e-8

if nonzero_mask.any():
    mape = np.mean(
        np.abs(errors[nonzero_mask] / actual[nonzero_mask])
    ) * 100
else:
    mape = np.nan

# ==========================================
# 6. TẠO BẢNG SO SÁNH
# ==========================================

comparison = pd.DataFrame({
    "Hour": np.arange(1, 25),
    "Actual_PM25": actual,
    "Predicted_PM25": forecast,
    "Absolute_Error": np.abs(errors)
})

comparison.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 7. VẼ BIỂU ĐỒ
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    comparison["Hour"],
    comparison["Actual_PM25"],
    marker="o",
    label="Actual PM2.5"
)

plt.plot(
    comparison["Hour"],
    comparison["Predicted_PM25"],
    marker="s",
    label="SOFTS API Forecast"
)

plt.xlabel("Forecast Hour")
plt.ylabel("PM2.5 (µg/m³)")
plt.title("SOFTS REST API: Actual vs Predicted PM2.5")
plt.xticks(np.arange(1, 25))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

# ==========================================
# 8. LƯU VÀ HIỂN THỊ KẾT QUẢ
# ==========================================

metrics_text = f"""
SOFTS REST API - FORECAST EVALUATION
====================================

Dataset: X_test[0] / Y_test[0]
Forecast horizon: 24 hours

MAE  : {mae:.6f} µg/m³
MSE  : {mse:.6f}
RMSE : {rmse:.6f} µg/m³
MAPE : {mape:.6f} %

Mean actual     : {np.mean(actual):.6f} µg/m³
Mean prediction : {np.mean(forecast):.6f} µg/m³

Minimum actual     : {np.min(actual):.6f}
Maximum actual     : {np.max(actual):.6f}
Minimum prediction : {np.min(forecast):.6f}
Maximum prediction : {np.max(forecast):.6f}
"""

OUTPUT_METRICS.write_text(
    metrics_text,
    encoding="utf-8"
)

print(metrics_text)

print("ĐÃ LƯU CÁC KẾT QUẢ:")
print(f"1. Bảng so sánh : {OUTPUT_CSV}")
print(f"2. Biểu đồ      : {OUTPUT_PLOT}")
print(f"3. Chỉ số       : {OUTPUT_METRICS}")

print("\nBẢNG SO SÁNH 24 GIỜ:")
print(comparison.round(4).to_string(index=False))
