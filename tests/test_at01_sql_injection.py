
import csv
import os
from pathlib import Path

import numpy as np
import pytest
import requests


# ============================================================
# CAU HINH
# ============================================================

BASE_URL = os.getenv(
    "SOFTS_TEST_API_URL",
    "http://127.0.0.1:8000"
)

API_KEY = os.getenv(
    "SOFTS_TEST_API_KEY",
    "at02-test-token"
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
# HAM HO TRO
# ============================================================

def load_test_input():
    """Lay mot mau du lieu hop le co kich thuoc 96 x 7."""

    if not DATA_FILE.exists():
        pytest.fail(f"Khong tim thay du lieu: {DATA_FILE}")

    dataset = np.load(DATA_FILE)

    if "X_test" not in dataset:
        pytest.fail("File NPZ khong chua X_test")

    values = dataset["X_test"][0].astype(float).tolist()

    assert len(values) == 96
    assert all(len(row) == 7 for row in values)

    return values


def count_history_records():
    """Dem so ban ghi trong file lich su du bao."""

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
    """Gui request kem API Key hop le."""

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    return requests.post(
        f"{BASE_URL}/forecast",
        json={"values": values},
        headers=headers,
        timeout=TIMEOUT
    )


# ============================================================
# AT01.1 - SQL INJECTION: OR 1=1
# ============================================================

def test_at01_1_sql_injection_or_condition():

    values = load_test_input()
    values[0][0] = "' OR '1'='1"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API khong tu choi payload OR 1=1. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lich su du bao bi thay doi sau yeu cau khong hop le"
    )


# ============================================================
# AT01.2 - SQL INJECTION: DROP TABLE
# ============================================================

def test_at01_2_sql_injection_drop_table():

    values = load_test_input()
    values[0][0] = "'; DROP TABLE forecast_history;--"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API khong tu choi payload DROP TABLE. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "So ban ghi lich su thay doi sau payload DROP TABLE"
    )


# ============================================================
# AT01.3 - SQL INJECTION: UNION SELECT
# ============================================================

def test_at01_3_sql_injection_union_select():

    values = load_test_input()
    values[0][0] = "1 UNION SELECT * FROM users--"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API khong tu choi payload UNION SELECT. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lich su du bao bi thay doi sau payload UNION SELECT"
    )


# ============================================================
# AT01.4 - SQL INJECTION: COMMENT
# ============================================================

def test_at01_4_sql_injection_comment():

    values = load_test_input()
    values[0][0] = "admin'/*"

    history_before = count_history_records()

    response = send_forecast_request(values)

    history_after = count_history_records()

    assert response.status_code == 422, (
        f"API khong tu choi payload SQL comment. "
        f"HTTP status: {response.status_code}"
    )

    assert history_after == history_before, (
        "Lich su du bao bi thay doi sau payload SQL comment"
    )


# ============================================================
# AT01.5 - KIEM TRA API SAU CAC YEU CAU TAN CONG
# ============================================================

def test_at01_5_api_health_after_injection():

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT
    )

    assert response.status_code == 200, (
        f"API khong phan hoi binh thuong. "
        f"HTTP status: {response.status_code}"
    )
