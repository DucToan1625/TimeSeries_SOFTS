
import numpy as np
import pandas as pd
from pathlib import Path
from api.main import predict

# ==========================================
# 1. ĐỌC DỮ LIỆU VÀ SCALER
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

data = np.load(BASE_DIR / "data" / "windows_96_to_24.npz")
scaler = pd.read_csv(BASE_DIR / "data" / "scaler_stats.csv")

X_test = data["X_test"]
Y_test = data["Y_test"]

pm25_scaler = scaler.loc[
    scaler["feature"] == "pm25"
].iloc[0]

PM25_MEAN = float(pm25_scaler["mean"])
PM25_SCALE = float(pm25_scaler["scale"])

print("Số mẫu kiểm thử:", len(X_test))
print("Kích thước đầu vào:", X_test.shape)
print("Kích thước đầu ra:", Y_test.shape)

# ==========================================
# 2. DỰ BÁO TOÀN BỘ X_TEST
# ==========================================

all_predictions = []

for i in range(len(X_test)):
    prediction = predict(X_test[i].tolist())
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)

    if prediction.shape != (24,):
        raise ValueError(
            f"Mẫu {i}: API trả về {prediction.shape}, "
            "không phải 24 giá trị"
        )

    if not np.isfinite(prediction).all():
        raise ValueError(f"Mẫu {i} có dự báo NaN/Inf")

    all_predictions.append(prediction)

    if (i + 1) % 50 == 0:
        print(f"Đã dự báo: {i + 1}/{len(X_test)}")

predictions = np.asarray(all_predictions)

# ==========================================
# 3. CHUYỂN Y_TEST VỀ ĐƠN VỊ GỐC
# ==========================================

actual = (
    Y_test.reshape(len(Y_test), 24)
    * PM25_SCALE
    + PM25_MEAN
)

# ==========================================
# 4. TÍNH CHỈ SỐ TOÀN TẬP
# ==========================================

errors = predictions - actual

mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)

mask = np.abs(actual) > 1e-8
mape = np.mean(
    np.abs(errors[mask] / actual[mask])
) * 100

print("\n===== FULL TEST EVALUATION =====")
print(f"Samples: {len(X_test)}")
print(f"Forecast values: {predictions.size}")
print(f"MAE : {mae:.6f} µg/m³")
print(f"MSE : {mse:.6f}")
print(f"RMSE: {rmse:.6f} µg/m³")
print(f"MAPE: {mape:.6f}%")

# ==========================================
# 5. LƯU KẾT QUẢ
# ==========================================

np.savez_compressed(
    BASE_DIR / "full_test_predictions.npz",
    actual=actual,
    predictions=predictions
)

pd.DataFrame({
    "sample": np.arange(len(X_test)),
    "MAE": np.mean(np.abs(errors), axis=1),
    "RMSE": np.sqrt(np.mean(errors ** 2, axis=1))
}).to_csv(
    BASE_DIR / "full_test_sample_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    BASE_DIR / "full_test_metrics.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("SOFTS FULL TEST EVALUATION\n")
    f.write(f"Samples: {len(X_test)}\n")
    f.write(f"Forecast values: {predictions.size}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAPE: {mape:.6f}%\n")

print("\nĐã lưu kết quả đánh giá toàn bộ tập test.")
