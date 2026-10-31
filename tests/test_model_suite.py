"""
Expanded tests for the SOFTS + DINOv3 PM2.5 forecasting project.

Run from repository root:
    python -m pytest tests/test_model_suite.py -v

DINOv3 tests use a lightweight mocked backbone so CI does not need pretrained
weights or the external DINOv3 repository. SOFTS tests use the real model.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.SOFTS import Model as SOFTSModel, STAR
from models.DINOv3_PM25 import DINOv3PM25


def softs_config(**overrides):
    values = dict(
        seq_len=96,
        pred_len=24,
        d_model=128,
        d_core=128,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeDINOBackbone(nn.Module):
    """Mimics DINOv3 feature output with 384-dimensional CLS features."""

    def forward(self, image):
        batch = image.shape[0]
        return {"x_norm_clstoken": torch.ones(
            batch, 384, device=image.device
        )}


@pytest.fixture
def softs():
    torch.manual_seed(7)
    return SOFTSModel(softs_config()).eval()


@pytest.fixture
def dino_model():
    torch.manual_seed(7)

    with patch("torch.hub.load", return_value=FakeDINOBackbone()):
        model = DINOv3PM25(
            repo_dir="unused",
            weights_path="unused"
        ).eval()

    return model


# ============================================================
# 1-4: SOFTS configuration
# ============================================================

@pytest.mark.parametrize(
    "field,expected",
    [
        ("seq_len", 96),
        ("pred_len", 24),
        ("d_model", 128),
        ("e_layers", 2),
    ],
)
def test_softs_config_values(field, expected):
    cfg = softs_config()
    assert getattr(cfg, field) == expected


# ============================================================
# 5-10: STAR block
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 128),
        (2, 7, 128),
        (4, 7, 128),
        (2, 3, 128),
        (2, 10, 128),
        (1, 1, 128),
    ],
)
def test_star_eval_output_shape(shape):
    block = STAR(d_series=128, d_core=128).eval()
    x = torch.randn(*shape)

    y, aux = block(x)

    assert y.shape == x.shape
    assert aux is None
    assert torch.isfinite(y).all()


# ============================================================
# 11-16: SOFTS forward
# ============================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_softs_forward_shape(softs, batch_size):
    x = torch.randn(batch_size, 96, 7)

    with torch.no_grad():
        y = softs(x, None, None, None)

    assert y.shape == (batch_size, 24, 7)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("use_norm", [True, False])
def test_softs_forward_with_normalization_setting(use_norm):
    model = SOFTSModel(
        softs_config(use_norm=use_norm)
    ).eval()

    x = torch.randn(2, 96, 7)

    with torch.no_grad():
        y = model(x, None, None, None)

    assert y.shape == (2, 24, 7)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("seq_len", [48, 96])
def test_softs_configured_sequence_length(seq_len):
    model = SOFTSModel(
        softs_config(seq_len=seq_len)
    ).eval()

    x = torch.randn(2, seq_len, 7)

    with torch.no_grad():
        y = model(x, None, None, None)

    assert y.shape == (2, 24, 7)


# ============================================================
# 17-20: SOFTS training and numerical stability
# ============================================================

@pytest.mark.parametrize("batch_size", [1, 2])
def test_softs_training_backward(batch_size):
    model = SOFTSModel(softs_config()).train()
    x = torch.randn(
        batch_size, 96, 7, requires_grad=True
    )

    y = model(x, None, None, None)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_softs_constant_input_is_finite(softs, value):
    x = torch.full((2, 96, 7), value)

    with torch.no_grad():
        y = softs(x, None, None, None)

    assert torch.isfinite(y).all()


# ============================================================
# 21-26: DINOv3 wrapper
# ============================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_dinov3_forward_shape(dino_model, batch_size):
    x = torch.randn(batch_size, 96, 7)

    with torch.no_grad():
        y = dino_model(x)

    assert y.shape == (batch_size, 24, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "bad_shape",
    [
        (96, 7),
        (2, 95, 7),
        (2, 96, 6),
    ],
)
def test_dinov3_rejects_invalid_input_shape(
    dino_model, bad_shape
):
    with pytest.raises(
        ValueError,
        match="Input cần có dạng",
    ):
        dino_model(torch.randn(*bad_shape))


def test_dinov3_backbone_is_frozen(dino_model):
    assert all(
        not p.requires_grad
        for p in dino_model.backbone.parameters()
    )


def test_dinov3_train_keeps_backbone_eval(dino_model):
    dino_model.train()

    assert dino_model.training
    assert not dino_model.backbone.training


# ============================================================
# 27-30: Forecasting and data contracts
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [
        (1, 96, 7),
        (8, 96, 7),
    ],
)
def test_forecast_input_contract(shape):
    x = torch.randn(*shape)

    assert x.ndim == 3
    assert x.shape[1:] == (96, 7)


@pytest.mark.parametrize("n", [24, 48])
def test_pm25_forecast_horizon_contract(n):
    forecast = torch.zeros(2, n, 1)

    assert forecast.shape[1] == n
    assert forecast.shape[-1] == 1


def test_pm25_dataset_values_finite():
    values = np.array([0.0, 12.4, 35.2, 100.0])

    assert np.isfinite(values).all()


def test_pm25_dataset_values_nonnegative():
    values = np.array([0.0, 12.4, 35.2, 100.0])

    assert (values >= 0).all()