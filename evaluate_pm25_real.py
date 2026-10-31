
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================================================
# 1. Khai báo đường dẫn
# ==================================================
root = Path(".")
checkpoint_dir = (
    root / "checkpoints" /
    "long_term_forecast_Hanoi_PM25_SOFTS_PM25_"
    "ftMS_sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_ebtimeF_dtTrue_test"
)

prediction_file = checkpoint_dir / "pm25_test_predictions.npz"
scaler_file = root / "data" / "scaler_stats.csv"

# ==================================================
# 2. Đọc dự báo và giá trị thực
# ==================================================
data = np.load(prediction_file)

predictions = data["predictions"].astype(np.float64)
targets = data["targets"].astype(np.float64)

print("Prediction shape:", predictions.shape)
print("Target shape:", targets.shape)

if predictions.shape != targets.shape:
    raise ValueError("Kích thước dự báo và giá trị thực không khớp!")

# ==================================================
# 3. Đọc thống kê StandardScaler của PM2.5
# ==================================================
scaler_df = pd.read_csv(scaler_file)
row = scaler_df.loc[scaler_df["feature"] == "pm25"]

if len(row) != 1:
    raise ValueError("Không tìm thấy đúng một dòng pm25 trong scaler_stats.csv")

mean = float(row.iloc[0]["mean"])
scale = float(row.iloc[0]["scale"])

print(f"\nPM2.5 mean  = {mean:.6f}")
print(f"PM2.5 scale = {scale:.6f}")

# ==================================================
# 4. Đánh giá trên thang chuẩn hóa
# ==================================================
pred_std = predictions.reshape(-1)
true_std = targets.reshape(-1)

mse_std = mean_squared_error(true_std, pred_std)
rmse_std = np.sqrt(mse_std)
mae_std = mean_absolute_error(true_std, pred_std)

# ==================================================
# 5. Chuyển về đơn vị PM2.5 thực tế
# ==================================================
pred_real = pred_std * scale + mean
true_real = true_std * scale + mean

mse_real = mean_squared_error(true_real, pred_real)
rmse_real = np.sqrt(mse_real)
mae_real = mean_absolute_error(true_real, pred_real)

# ==================================================
# 6. In kết quả
# ==================================================
print("\n========== KẾT QUẢ ĐÁNH GIÁ SOFTS ==========")
print("Số mẫu test:", predictions.shape[0])
print("Số giờ dự báo mỗi mẫu:", predictions.shape[1])

print("\n--- Thang chuẩn hóa ---")
print(f"MSE  : {mse_std:.6f}")
print(f"RMSE : {rmse_std:.6f}")
print(f"MAE  : {mae_std:.6f}")

print("\n--- Thang PM2.5 thực tế (µg/m³) ---")
print(f"MSE  : {mse_real:.6f} (µg/m³)²")
print(f"RMSE : {rmse_real:.6f} µg/m³")
print(f"MAE  : {mae_real:.6f} µg/m³")

print("\n--- Ví dụ 5 giá trị đầu tiên ---")
for i in range(min(5, len(pred_real))):
    print(
        f"Giá trị thực: {true_real[i]:.3f} | "
        f"Dự báo: {pred_real[i]:.3f} µg/m³"
    )

# ==================================================
# 7. Lưu kết quả đánh giá
# ==================================================
result = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE"],
    "Standardized": [mse_std, rmse_std, mae_std],
    "Original_PM25": [mse_real, rmse_real, mae_real],
    "Unit": ["(µg/m³)²", "µg/m³", "µg/m³"]
})

output_file = root / "pm25_evaluation_results.csv"
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\nĐã lưu kết quả tại: {output_file.resolve()}")
