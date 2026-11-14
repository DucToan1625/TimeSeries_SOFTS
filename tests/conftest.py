
import importlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


# Thư mục gốc của dự án SOFTS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API key dùng riêng cho kiểm thử
TEST_API_KEY = "softs-test-secret-2026"


@pytest.fixture(scope="session")
def test_data():
    """
    Đọc dữ liệu kiểm thử trực tiếp từ windows_96_to_24.npz.
    Sử dụng mẫu X_test[0], không tự tạo dữ liệu giả.
    """
    data_path = PROJECT_ROOT / "data" / "windows_96_to_24.npz"

    if not data_path.exists():
        pytest.fail(f"Không tìm thấy dữ liệu: {data_path}")

    dataset = np.load(data_path)

    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    assert X_test.ndim == 3
    assert X_test.shape[1:] == (96, 7)
    assert Y_test.shape[1:] == (24, 1)

    return {
        "input": X_test[0].astype(np.float32),
        "actual": Y_test[0].astype(np.float32),
        "X_test_shape": X_test.shape,
        "Y_test_shape": Y_test.shape,
    }


@pytest.fixture
def api_module(monkeypatch):
    """
    Nạp API và thiết lập API key cho môi trường kiểm thử.
    """
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    module = importlib.import_module("api.main")

    # Xóa lịch sử rate limit giữa các test
    if hasattr(module, "_request_times"):
        module._request_times.clear()

    return module


@pytest.fixture
def client(api_module):
    """
    Tạo HTTP test client để gọi API trực tiếp,
    không cần khởi chạy Uvicorn.
    """
    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload(test_data):
    """
    Chuyển mẫu dữ liệu X_test[0] thành JSON request.
    """
    return {
        "values": test_data["input"].tolist()
    }


@pytest.fixture
def auth_headers():
    """
    Header xác thực API.
    """
    return {
        "Authorization": f"Bearer {TEST_API_KEY}"
    }
