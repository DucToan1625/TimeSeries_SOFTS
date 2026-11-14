import csv
import os
from pathlib import Path

import numpy as np
import pytest
import requests


# ============================================================
# CẤU HÌNH
# ============================================================

BASE_URL = os.getenv(
    "SOFTS_TEST_API_URL",
    "http://127.0.0.1:8000"
)

TIMEOUT = float(
    os.getenv("SOFTS_TEST_TIMEOUT", "30")
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "windows_96_to_24.npz"

HISTORY_FILE = Path(
    os.getenv(
        "FORECAST_HISTORY_FILE",
        str(PROJECT_ROOT / "data" / "forecast_history.csv")
    )
)


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def load_test_input():
    """
    Lấy một mẫu dữ liệu hợp lệ từ tập X_test.
    Đầu vào có kích thước 96 x 7.
    """

    if not DATA_FILE.exists():
        pytest.fail(f"Không tìm thấy dữ liệu: {DATA_FILE}")

    dataset = np.load(DATA_FILE)

    if "X_test" not in dataset:
        pytest.fail("File NPZ không chứa X_test")

    values = dataset["X_test"][0].astype(float).tolist()

    assert len(values) == 96
    assert all(len(row) == 7 for row in values)

    return values


def count_history_records():
    """
    Đếm số bản ghi trong file lịch sử dự báo.
    Không tính dòng tiêu đề.
    """

    if not HISTORY_FILE.exists():
        return 0

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return max(sum(1 for _ in csv.reader(file)) - 1, 0)


def send_forecast_request(values):
    """
    Gửi dữ liệu kiểm thử tới API /forecast.
    """

    return requests.post(
        f"{BASE_URL}/forecast",
        json={"values": values},
        timeout=TIMEOUT
    )


# ============================================================
# AT01.1 - SQL INJECTION: OR 1=1
# ============================================================

def test_at01_1_sql_injection_or_condition():
    """
    Kiểm tra payload SQL Injection dạng OR 1=1.
    API phải từ chối dữ liệu không đúng kiểu.
    """

    values = load_test_input()

    values[0][0] = "' OR '1'='1"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API không từ chối payload OR 1=1. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lịch sử dự báo bị thay đổi sau yêu cầu không hợp lệ"
    )


# ============================================================
# AT01.2 - SQL INJECTION: DROP TABLE
# ============================================================

def test_at01_2_sql_injection_drop_table():
    """
    Kiểm tra payload có ý đồ xóa bảng dữ liệu.
    """

    values = load_test_input()

    values[0][0] = "'; DROP TABLE forecast_history;--"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API không từ chối payload DROP TABLE. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Số bản ghi lịch sử thay đổi sau payload DROP TABLE"
    )


# ============================================================
# AT01.3 - SQL INJECTION: UNION SELECT
# ============================================================

def test_at01_3_sql_injection_union_select():
    """
    Kiểm tra payload UNION SELECT.
    """

    values = load_test_input()

    values[0][0] = "1 UNION SELECT * FROM users--"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API không từ chối payload UNION SELECT. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lịch sử dự báo bị thay đổi sau payload UNION SELECT"
    )


# ============================================================
# AT01.4 - SQL INJECTION: COMMENT
# ============================================================

def test_at01_4_sql_injection_comment():
    """
    Kiểm tra payload sử dụng ký tự comment trong SQL.
    """

    values = load_test_input()

    values[0][0] = "admin'/*"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API không từ chối payload SQL comment. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lịch sử dự báo bị thay đổi sau payload SQL comment"
    )


# ============================================================
# AT01.5 - KIỂM TRA API SAU CÁC YÊU CẦU TẤN CÔNG
# ============================================================

def test_at01_5_api_health_after_injection():
    """
    Kiểm tra API vẫn phản hồi sau khi nhận các payload không hợp lệ.
    """

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT
    )

    assert response.status_code == 200, (
        f"API không phản hồi bình thường. "
        f"HTTP status: {response.status_code}"
    )