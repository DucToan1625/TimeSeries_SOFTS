
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================
ROOT = Path(".")
DATA_DIR = ROOT / "data"

RESULT_DIR = (
    ROOT / "checkpoints" /
    "long_term_forecast_Hanoi_PM25_SOFTS_PM25_"
    "ftMS_sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_ebtimeF_dtTrue_test"
)

PREDICTION_FILE = RESULT_DIR / "pm25_test_predictions.npz"
SCALER_FILE = DATA_DIR / "scaler_stats.csv"

OUTPUT_FILE = ROOT / "softs_pm25_96h_24h.png"

# =====================================================
# 2. THIẾT LẬP
# =====================================================
HISTORY_HOURS = 96
FORECAST_HOURS = 24

# Chọn mẫu test muốn hiển thị
SAMPLE_INDEX = 0

# =====================================================
# 3. ĐỌC KẾT QUẢ DỰ BÁO
# =====================================================
data = np.load(PREDICTION_FILE)

predictions = data["predictions"]  # (1014, 24, 1)
targets = data["targets"]          # (1014, 24, 1)

predictions = predictions[:, :, 0]
targets = targets[:, :, 0]

if not 0 <= SAMPLE_INDEX < len(predictions):
    raise IndexError("SAMPLE_INDEX nằm ngoài phạm vi dữ liệu test.")

pred_scaled = predictions[SAMPLE_INDEX]
true_future_scaled = targets[SAMPLE_INDEX]

# =====================================================
# 4. ĐỌC THỐNG KÊ STANDARD SCALER
# =====================================================
scaler_df = pd.read_csv(SCALER_FILE)
pm25_row = scaler_df.loc[scaler_df["feature"] == "pm25"]

if len(pm25_row) != 1:
    raise ValueError("Không tìm thấy thống kê chuẩn hóa của PM2.5.")

mean = float(pm25_row.iloc[0]["mean"])
scale = float(pm25_row.iloc[0]["scale"])

# Chuyển dự báo và nhãn về đơn vị thực tế
pred_real = pred_scaled * scale + mean
true_future_real = true_future_scaled * scale + mean

# =====================================================
# 5. LẤY 96 GIỜ LỊCH SỬ TỪ DỮ LIỆU TEST
# =====================================================
# NPZ hiện tại chỉ lưu predictions và targets.
# Để lấy lịch sử đúng với cửa sổ test, đọc X_test từ file dữ liệu.
dataset_file = DATA_DIR / "windows_96_to_24.npz"
dataset = np.load(dataset_file)

if "X_test" not in dataset.files:
    raise KeyError("Không tìm thấy X_test trong windows_96_to_24.npz.")

X_test = dataset["X_test"]

if X_test.ndim != 3 or X_test.shape[1:] != (96, 7):
    raise ValueError(f"Shape X_test không hợp lệ: {X_test.shape}")

# Feature PM2.5 nằm ở vị trí 0
history_scaled = X_test[SAMPLE_INDEX, :, 0]
history_real = history_scaled * scale + mean

# =====================================================
# 6. TẠO TRỤC THỜI GIAN TƯƠNG ĐỐI
# =====================================================
# 96 giờ trước hiện tại: -95 đến 0
# 24 giờ dự báo: 1 đến 24
history_x = np.arange(-HISTORY_HOURS + 1, 1)
forecast_x = np.arange(1, FORECAST_HOURS + 1)

# =====================================================
# 7. VẼ BIỂU ĐỒ
# =====================================================
plt.figure(figsize=(15, 6))

# Vùng lịch sử và vùng dự báo
plt.axvspan(
    -HISTORY_HOURS + 1, 0,
    color="#2563eb", alpha=0.045
)
plt.axvspan(
    1, FORECAST_HOURS,
    color="#f59e0b", alpha=0.09
)

# Dữ liệu lịch sử
plt.plot(
    history_x,
    history_real,
    color="#2563eb",
    linewidth=2,
    label="Historical PM2.5 (96 h)"
)

# Nối điểm cuối lịch sử với điểm dự báo đầu tiên
plt.plot(
    [0, 1],
    [history_real[-1], pred_real[0]],
    color="#f59e0b",
    linewidth=2,
    linestyle="--"
)

# Dự báo tương lai
plt.plot(
    forecast_x,
    pred_real,
    color="#f59e0b",
    linewidth=2,
    marker="o",
    markersize=3.5,
    label="SOFTS forecast (24 h)"
)

# Giá trị thực tương lai để đối chiếu
plt.plot(
    forecast_x,
    true_future_real,
    color="#16a34a",
    linewidth=1.7,
    linestyle="--",
    marker=".",
    markersize=4,
    label="Actual PM2.5 (test)"
)

# Ranh giới hiện tại
plt.axvline(
    x=0,
    color="#475569",
    linestyle=":",
    linewidth=1.5,
    label="Forecast starts here"
)

# =====================================================
# 8. NGƯỠNG THAM KHẢO
# =====================================================
# Có thể chỉnh hoặc bỏ ngưỡng theo tiêu chuẩn dự án.
PM25_REFERENCE = 50

plt.axhline(
    y=PM25_REFERENCE,
    color="#64748b",
    linestyle="--",
    linewidth=1.2,
    label=f"Reference: {PM25_REFERENCE} µg/m³"
)

# =====================================================
# 9. THÔNG TIN DỰ BÁO
# =====================================================
peak_idx = int(np.argmax(pred_real))
peak_value = float(pred_real[peak_idx])
peak_hour = int(forecast_x[peak_idx])

plt.scatter(
    peak_hour,
    peak_value,
    color="#f59e0b",
    edgecolors="white",
    linewidths=1,
    s=85,
    zorder=5
)

plt.annotate(
    f"Peak: {peak_value:.1f} µg/m³\n"
    f"+{peak_hour} h",
    xy=(peak_hour, peak_value),
    xytext=(-10, 15),
    textcoords="offset points",
    ha="right",
    fontsize=9,
    fontweight="bold"
)

plt.title(
    "SOFTS Air Pollution Prediction: "
    "96-Hour History + 24-Hour Forecast",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Hours relative to forecast start")
plt.ylabel("PM2.5 (µg/m³)")

plt.xticks(
    [-95, -72, -48, -24, 0, 6, 12, 18, 24],
    ["−95 h", "−72 h", "−48 h", "−24 h",
     "Now", "+6 h", "+12 h", "+18 h", "+24 h"]
)

plt.grid(True, alpha=0.22)
plt.legend(loc="upper left", ncol=2, frameon=True)
plt.tight_layout()

# =====================================================
# 10. LƯU VÀ HIỂN THỊ
# =====================================================
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print("\n========== SOFTS FORECAST ==========")
print(f"Sample test       : {SAMPLE_INDEX}")
print(f"Historical points : {len(history_real)}")
print(f"Forecast points   : {len(pred_real)}")
print(f"Peak PM2.5        : {peak_value:.3f} µg/m³")
print(f"Peak forecast     : +{peak_hour} hours")
print(f"Saved chart       : {OUTPUT_FILE.resolve()}")
