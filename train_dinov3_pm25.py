
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.DINOv3_PM25 import DINOv3PM25


# =========================
# 1. Cấu hình
# =========================
SEED = 42
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 1e-3
PATIENCE = 5

DATA_PATH = "data/windows_96_to_24.npz"
SAVE_DIR = "checkpoints/dinov3_pm25"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# 2. Đọc dữ liệu
# =========================
data = np.load(DATA_PATH)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
Y_train = torch.tensor(data["Y_train"], dtype=torch.float32)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
Y_val = torch.tensor(data["Y_val"], dtype=torch.float32)

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
Y_test = torch.tensor(data["Y_test"], dtype=torch.float32)

print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)

train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, Y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# 3. Khởi tạo mô hình
# =========================
model = DINOv3PM25(
    forecast_hours=24,
    dropout=0.1
).to(device)

# Chỉ tối ưu các tham số được phép huấn luyện.
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

criterion = nn.MSELoss()

best_val_loss = float("inf")
best_state = None
patience_count = 0


# =========================
# 4. Huấn luyện
# =========================
for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)
        train_count += xb.size(0)

    train_loss = train_loss_sum / train_count

    # Đánh giá validation sau mỗi epoch.
    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            val_loss_sum += loss.item() * xb.size(0)
            val_count += xb.size(0)

    val_loss = val_loss_sum / val_count

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    # Lưu mô hình tốt nhất theo validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_count = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1
            },
            os.path.join(SAVE_DIR, "best_model.pth")
        )

        print("  -> Đã lưu checkpoint tốt nhất.")
    else:
        patience_count += 1

    if patience_count >= PATIENCE:
        print("Early stopping.")
        break


# =========================
# 5. Đánh giá trên tập test
# =========================
checkpoint = torch.load(
    os.path.join(SAVE_DIR, "best_model.pth"),
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

predictions = []
actuals = []

with torch.inference_mode():
    for start in range(0, len(X_test), BATCH_SIZE):
        xb = X_test[start:start + BATCH_SIZE].to(device)
        yb = Y_test[start:start + BATCH_SIZE].to(device)

        pred = model(xb)

        predictions.append(pred.cpu().numpy())
        actuals.append(yb.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Dữ liệu PM2.5 được chuẩn hóa bằng mean và scale
# của cột pm25 trong scaler_stats.csv.
stats = np.genfromtxt(
    "data/scaler_stats.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8"
)

pm25_row = next(row for row in stats if row["feature"] == "pm25")
pm25_mean = float(pm25_row["mean"])
pm25_scale = float(pm25_row["scale"])

# Đổi về đơn vị gốc µg/m³.
pred_original = predictions * pm25_scale + pm25_mean
actual_original = actuals * pm25_scale + pm25_mean

mse = np.mean((pred_original - actual_original) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred_original - actual_original))

print("\n========== TEST RESULTS ==========")
print("Best epoch:", checkpoint["epoch"])
print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
print(f"Test MSE:  {mse:.4f} (µg/m³)²")
print(f"Test RMSE: {rmse:.4f} µg/m³")
print(f"Test MAE:  {mae:.4f} µg/m³")

np.savez(
    os.path.join(SAVE_DIR, "test_predictions.npz"),
    predictions=predictions,
    actuals=actuals,
    predictions_original=pred_original,
    actuals_original=actual_original
)

print("\nĐã lưu mô hình và kết quả dự báo tại:", SAVE_DIR)
