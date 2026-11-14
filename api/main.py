
"""FastAPI wrapper for the repository's existing SOFTS forecaster.

The API loads the same SOFTS architecture/checkpoint/scaler as the Streamlit app.
Set SOFTS_ROOT if running outside the repository root.
Set API_KEY to enable Bearer-token authentication.
Forecast results are stored in data/forecast_history.csv.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from typing import Annotated

import numpy as np
import pandas as pd
import torch

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)

from api.storage import save_forecast


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(
    os.getenv(
        "SOFTS_ROOT",
        Path(__file__).resolve().parents[1]
    )
).resolve()

SCALER_FILE = ROOT / "data" / "scaler_stats.csv"

MODEL_FILE = ROOT / "checkpoints" / (
    "long_term_forecast_Hanoi_PM25_SOFTS_PM25_ftMS_sl96_ll48_"
    "pl24_dm128_el2_dl1_df256_fc1_ebtimeF_dtTrue_test"
) / "checkpoint.pth"

FEATURES = [
    "pm25",
    "pm10",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "rainfall"
]

INPUT_HOURS = 96
OUTPUT_HOURS = 24

RATE_LIMIT = int(
    os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")
)


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hanoi PM2.5 Forecast API",
    version="1.0.0"
)


# ============================================================
# 3. GLOBAL MODEL RESOURCES
# ============================================================

_model = None
_device = None
_scaler = None
_load_error: str | None = None

_load_lock = Lock()
_rate_lock = Lock()

_request_times: dict[str, list[float]] = {}


# ============================================================
# 4. SAFE VALIDATION ERROR HANDLER
# ============================================================

@app.exception_handler(RequestValidationError)
async def safe_validation_error_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Return validation errors without exposing raw request bodies.
    """

    errors = [
        {
            "loc": list(item.get("loc", ())),
            "msg": str(item.get("msg", "Invalid input")),
            "type": str(item.get("type", "value_error"))
        }
        for item in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )


# ============================================================
# 5. REQUEST AND RESPONSE SCHEMAS
# ============================================================

class ForecastRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    values: list[list[float]] = Field(
        ...,
        description="96 rows x 7 columns in FEATURE order"
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):

        arr = np.asarray(
            values,
            dtype=np.float64
        )

        # Validate input dimensions
        if arr.shape != (
            INPUT_HOURS,
            len(FEATURES)
        ):
            raise ValueError(
                f"values must have shape (96, 7); "
                f"received {arr.shape}"
            )

        # Reject NaN and infinity
        if not np.isfinite(arr).all():
            raise ValueError(
                "values must contain only finite numbers"
            )

        # Validate maximum input magnitude
        max_abs = float(
            os.getenv(
                "API_MAX_ABS_INPUT",
                "1000000"
            )
        )

        if np.max(np.abs(arr)) > max_abs:
            raise ValueError(
                f"input magnitude exceeds configured limit "
                f"({max_abs:g})"
            )

        return values


class ForecastResponse(BaseModel):

    forecast_pm25: list[float]

    horizon_hours: int = OUTPUT_HOURS

    unit: str = "ug/m3"

    model: str = "SOFTS"


# ============================================================
# 6. API KEY AUTHENTICATION
# ============================================================

def require_api_key(
    authorization: Annotated[
        str | None,
        Header()
    ] = None
):
    """
    Enable Bearer-token authentication when API_KEY is configured.
    """

    expected = os.getenv("API_KEY")

    if expected:

        supplied = authorization or ""

        if supplied != f"Bearer {expected}":

            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API token"
            )


# ============================================================
# 7. RATE LIMITING
# ============================================================

def enforce_rate_limit(request: Request):
    """
    Limit the number of valid requests from each client per minute.
    """

    client = (
        request.client.host
        if request.client
        else "unknown"
    )

    now = time.time()

    with _rate_lock:

        recent = [
            t
            for t in _request_times.get(client, [])
            if now - t < 60
        ]

        if len(recent) >= RATE_LIMIT:

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        recent.append(now)

        _request_times[client] = recent


# ============================================================
# 8. LOAD SOFTS MODEL AND SCALER
# ============================================================

def load_resources():

    global _model
    global _device
    global _scaler
    global _load_error

    if _model is not None and _scaler is not None:
        return

    with _load_lock:

        if _model is not None and _scaler is not None:
            return

        try:

            # Check model and scaler files
            if (
                not SCALER_FILE.is_file()
                or not MODEL_FILE.is_file()
            ):

                raise FileNotFoundError(
                    "Missing deployment artifacts: "
                    "data/scaler_stats.csv and/or SOFTS checkpoint"
                )

            from types import SimpleNamespace

            from models.SOFTS import Model

            # Load scaler
            df = pd.read_csv(SCALER_FILE)

            df.columns = [
                str(c).strip().lower()
                for c in df.columns
            ]

            fcol = next(
                (
                    c
                    for c in (
                        "feature",
                        "column",
                        "variable",
                        "name"
                    )
                    if c in df
                ),
                None
            )

            mcol = next(
                (
                    c
                    for c in ("mean", "mu")
                    if c in df
                ),
                None
            )

            scol = next(
                (
                    c
                    for c in (
                        "scale",
                        "std",
                        "standard_deviation"
                    )
                    if c in df
                ),
                None
            )

            if not all((fcol, mcol, scol)):

                raise ValueError(
                    "Scaler file must contain "
                    "feature, mean and scale/std columns"
                )

            stats = {}

            for _, row in df.iterrows():

                name = str(
                    row[fcol]
                ).strip().lower()

                mean = float(row[mcol])
                scale = float(row[scol])

                if (
                    np.isfinite(mean)
                    and np.isfinite(scale)
                    and scale > 0
                ):

                    stats[name] = {
                        "mean": mean,
                        "std": scale
                    }

            missing = [
                f
                for f in FEATURES
                if f not in stats
            ]

            if missing:

                raise ValueError(
                    f"Scaler is missing features: {missing}"
                )

            # SOFTS model configuration
            cfg = SimpleNamespace(

                task_name="long_term_forecast",

                enc_in=7,
                dec_in=1,
                c_out=1,

                seq_len=96,
                label_len=48,
                pred_len=24,

                d_model=128,
                d_core=128,

                n_heads=8,

                e_layers=2,
                d_layers=1,

                d_ff=256,

                dropout=0.1,
                fc_dropout=0.0,
                head_dropout=0.0,

                individual=False,

                embed="timeF",
                freq="h",

                activation="gelu",
                output_attention=False,

                use_norm=True,

                moving_avg=25,
                factor=1,

                distil=True,
                channel_independence=0,

                decomp_method="moving_avg",
                top_k=5,
                num_kernels=6,

                modes=32,
                mode_select="random",
                version="Fourier",

                L=3,
                base="legendre",
                cross_activation="tanh",

                patch_len=16,
                stride=8,
                padding_patch="end",

                revin=1,
                affine=0,
                subtract_last=0,

                kernel_size=25
            )

            # Select device
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = Model(cfg).to(device)

            # Load checkpoint
            try:

                checkpoint = torch.load(
                    MODEL_FILE,
                    map_location=device,
                    weights_only=False
                )

            except TypeError:

                checkpoint = torch.load(
                    MODEL_FILE,
                    map_location=device
                )

            if isinstance(checkpoint, dict):

                state = checkpoint.get(
                    "model",
                    checkpoint.get(
                        "state_dict",
                        checkpoint.get(
                            "model_state_dict",
                            checkpoint
                        )
                    )
                )

            else:

                state = checkpoint

            # Remove DataParallel prefix if present
            state = {
                (
                    k[7:]
                    if k.startswith("module.")
                    else k
                ): v
                for k, v in state.items()
            }

            model.load_state_dict(
                state,
                strict=True
            )

            model.eval()

            _model = model
            _device = device
            _scaler = stats
            _load_error = None

        except Exception as exc:

            _load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            raise


# ============================================================
# 9. SOFTS INFERENCE
# ============================================================

def predict(
    values: list[list[float]]
) -> list[float]:
    """
    Execute SOFTS inference and return 24 PM2.5 values.
    """

    load_resources()

    x = torch.as_tensor(
        np.asarray(
            values,
            dtype=np.float32
        )[None, ...],
        device=_device
    )

    decoder = torch.zeros(
        (1, 48 + 24, 1),
        dtype=torch.float32,
        device=_device
    )

    decoder[:, :48, 0] = x[:, -48:, 0]

    with torch.inference_mode():

        out = _model(
            x,
            None,
            decoder,
            None
        )

    if isinstance(out, (tuple, list)):
        out = out[0]

    if isinstance(out, dict):

        out = out.get(
            "prediction",
            out.get("output")
        )

    if (
        out is None
        or out.ndim != 3
        or out.shape[1] < 24
        or out.shape[2] not in (1, 7)
    ):

        raise RuntimeError(
            "Model returned an unexpected output shape"
        )

    channel = 0

    scaled = (
        out[0, -24:, channel]
        .detach()
        .cpu()
        .numpy()
    )

    # Inverse scaling
    result = (
        scaled * _scaler["pm25"]["std"]
        + _scaler["pm25"]["mean"]
    )

    # PM2.5 cannot be negative
    result = np.maximum(
        result,
        0.0
    )

    if (
        result.shape != (24,)
        or not np.isfinite(result).all()
    ):

        raise RuntimeError(
            "Model returned invalid forecast values"
        )

    return result.astype(float).tolist()


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    ready = (
        _model is not None
        and _scaler is not None
    )

    return {
        "status": (
            "ready"
            if ready
            else "not_loaded"
        ),
        "model": "SOFTS",
        "artifacts_present": (
            SCALER_FILE.is_file()
            and MODEL_FILE.is_file()
        )
    }


# ============================================================
# 11. FORECAST API WITH CSV STORAGE
# ============================================================

@app.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[
        Depends(require_api_key)
    ]
)
def forecast(
    payload: ForecastRequest,
    request: Request
):

    # Step 1: Apply rate limit after request validation
    enforce_rate_limit(request)

    # Step 2: Execute model inference
    try:

        values = predict(
            payload.values
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail="Forecast model is not configured"
        ) from exc

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Forecast inference failed"
        ) from exc

    # Step 3: Save forecast result to CSV
    try:

        save_forecast(values)

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail="Forecast storage unavailable"
        ) from exc

    # Step 4: Return forecast response
    return ForecastResponse(
        forecast_pm25=values
    )
