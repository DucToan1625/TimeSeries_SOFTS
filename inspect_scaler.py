
import pandas as pd
from pathlib import Path

file_path = Path("data/scaler_stats.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file: {file_path.resolve()}"
    )

df = pd.read_csv(file_path)

print("Các cột:", df.columns.tolist())
print("\nThông tin chuẩn hóa:")
print(df.to_string(index=False))
