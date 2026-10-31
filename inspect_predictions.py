
import numpy as np
from pathlib import Path

# Đường dẫn file kết quả dự báo
file_path = Path(
    "checkpoints/long_term_forecast_Hanoi_PM25_SOFTS_PM25_"
    "ftMS_sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_ebtimeF_dtTrue_test/"
    "pm25_test_predictions.npz"
)

# Đọc các mảng đã lưu
data = np.load(file_path)

print("Các mảng trong file:", data.files)

for name in data.files:
    arr = data[name]
    print(f"\n{name}")
    print("  Shape:", arr.shape)
    print("  Dtype:", arr.dtype)
    print("  Giá trị mẫu:", arr.reshape(-1)[:5])
