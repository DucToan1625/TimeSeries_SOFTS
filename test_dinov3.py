from pathlib import Path

import pytest
import torch


DINOV3_HUBCONF = Path("external/dinov3/hubconf.py")
CHECKPOINT = Path(
    "checkpoints/dinov3/"
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)


@pytest.mark.skipif(
    not DINOV3_HUBCONF.exists(),
    reason="DINOv3 source is not available",
)
@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="DINOv3 checkpoint is not available",
)
def test_dinov3_pretrained_forward():
    # Import và khởi tạo model bên trong test,
    # không chạy khi pytest đang collection.
    model = torch.hub.load(
        str(DINOV3_HUBCONF.parent),
        "dinov3_vits16",
        source="local",
        weights=str(CHECKPOINT),
    ).eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    if isinstance(output, dict):
        assert "x_norm_clstoken" in output
        output = output["x_norm_clstoken"]

    assert output.shape == (1, 384)
    assert torch.isfinite(output).all()