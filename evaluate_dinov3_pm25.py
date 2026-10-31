import numpy as np

path = "checkpoints/dinov3_pm25/test_predictions.npz"

data = np.load(path)

print("Các mảng:", data.files)

pred = data["predictions_original"]
actual = data["actuals_original"]

mse = np.mean((pred - actual) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred - actual))

print("\n===== DINOv3 - TEST RESULTS =====")
print("Prediction shape:", pred.shape)
print("Actual shape:", actual.shape)
print(f"MSE:  {mse:.4f} (µg/m³)²")
print(f"RMSE: {rmse:.4f} µg/m³")
print(f"MAE:  {mae:.4f} µg/m³")
