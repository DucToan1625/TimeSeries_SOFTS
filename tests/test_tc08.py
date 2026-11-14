
from fastapi.testclient import TestClient

import api.main as api

client = TestClient(api.app)


def test_tc08_model_inference_failure(monkeypatch):
    """TC08: Giả lập mô hình phát sinh lỗi khi dự báo."""

    def mock_predict(values):
        raise RuntimeError("Simulated model inference failure")

    monkeypatch.setattr(api, "predict", mock_predict)

    values = [[1.0 for _ in range(7)] for _ in range(96)]

    response = client.post(
        "/forecast",
        json={"values": values}
    )

    assert response.status_code == 500

    result = response.json()

    assert result["detail"] == "Forecast inference failed"

    # Không để lộ thông tin lỗi nội bộ của mô hình.
    assert "Simulated model inference failure" not in response.text
