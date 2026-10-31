from pathlib import Path

import pytest
import torch

DINOV3_HUBCONF = Path("external/dinov3/hubconf.py")


@pytest.mark.skipif(
    not DINOV3_HUBCONF.exists(),
    reason="DINOv3 source is not available",
)
def test_dinov3_pm25_forward():
    from models.DINOv3_PM25 import DINOv3PM25

    model = DINOv3PM25().eval()

    # Model nhận chuỗi 96 giờ với 7 biến đầu vào,
    # không nhận ảnh RGB trực tiếp.
    x = torch.randn(2, 96, 7)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 24, 1)
    assert torch.isfinite(y).all()