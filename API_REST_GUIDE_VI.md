# REST API SOFTS PM2.5 — Hướng dẫn triển khai

## Phạm vi giai đoạn này
Chỉ xây dựng và khởi chạy REST API. Chưa triển khai bộ kiểm thử. Phần kiểm thử sẽ được thực hiện ở giai đoạn sau với từng test case, gồm mục tiêu, tiền điều kiện, từng bước thực hiện, kết quả mong đợi, lỗi có thể gặp và cách xử lý.

## 1. Cấu trúc file được bổ sung
- `api/main.py`: FastAPI, gồm `GET /health` và `POST /forecast`.
- `api/client.py`: hàm client để Streamlit gọi REST API.
- `requirements-api.txt`: thư viện cần thiết cho API.
- `softs_pm25_deploy/app.py`: tích hợp tùy chọn gọi API qua biến môi trường `SOFTS_API_URL`; nếu không đặt biến này thì giữ cơ chế dự báo local.

API sử dụng checkpoint và scaler có sẵn trong repository. Đầu vào dự báo là 96 mốc thời gian × 7 đặc trưng theo thứ tự `pm25, pm10, temperature, humidity, wind_speed, pressure, rainfall`; đầu ra gồm 24 giá trị PM2.5.

## 2. Giải nén
Giải nén ZIP vào thư mục gốc repository hiện tại, giữ nguyên cấu trúc thư mục và cho phép ghi đè `softs_pm25_deploy/app.py`. Không xóa các thư mục `models`, `data`, `checkpoints` hiện có.

## 3. Cài thư viện (PowerShell)
Mở PowerShell tại thư mục gốc repository:

```powershell
pip install -r requirements-api.txt
```

## 4. Khởi chạy REST API
Trong PowerShell thứ nhất:

```powershell
$env:API_KEY = "thay-bang-chuoi-bi-mat-dai-va-ngau-nhien"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Giữ cửa sổ này mở. API chạy cục bộ tại `http://127.0.0.1:8000`. Tài liệu endpoint tự sinh tại `http://127.0.0.1:8000/docs`.

## 5. Kết nối Streamlit với REST API (tùy chọn)
Mở PowerShell thứ hai, tại cùng thư mục repository:

```powershell
$env:SOFTS_API_URL = "http://127.0.0.1:8000"
$env:API_KEY = "chuoi-bi-mat-giong-voi-terminal-api"
streamlit run softs_pm25_deploy/app.py
```

Nếu không đặt `SOFTS_API_URL`, ứng dụng tiếp tục dự báo bằng mô hình local như trước. Lưu ý: app hiện vẫn khởi tạo mô hình local khi khởi động theo cấu trúc ban đầu, kể cả khi bật API mode.

## 6. Lưu ý vận hành
- Chỉ bind `127.0.0.1` khi chạy thử trên máy cá nhân. Không đưa API ra Internet nếu chưa thiết lập HTTPS, reverse proxy/firewall, xác thực và giới hạn truy cập phù hợp.
- Không đưa API key thật vào mã nguồn hoặc Git.
- API hiện dùng rate limit lưu trong RAM, phù hợp chạy demo một tiến trình; triển khai nhiều worker/môi trường production cần chuyển sang bộ đếm dùng chung như Redis hoặc API gateway.
- Giai đoạn này chưa chạy hay bàn giao bộ test. Các bước kiểm thử và hướng xử lý lỗi sẽ được viết thành tài liệu riêng ở giai đoạn tiếp theo.
