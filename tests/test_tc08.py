
import os

from fastapi.testclient import TestClient

import api.main as api


TEST_API_KEY = os.getenv(
    "API_KEY",
    "softs-test-secret-2026"
)

client = TestClient(
    api.app,
    headers={
        "Authorization": f"Bearer {TEST_API_KEY}"
    }
)


def test_tc08_model_inference_failure(monkeypatch):
    """TC08: Gia lap mo hinh phat sinh loi khi du bao."""

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

    # Khong de lo thong tin loi noi bo cua mo hinh.
    assert "Simulated model inference failure" not in response.text
