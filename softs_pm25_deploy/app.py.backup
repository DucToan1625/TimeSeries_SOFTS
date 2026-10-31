
import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hanoi Air Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #F3F6FA !important;
        color: #183247 !important;
    }

    .block-container {
        max-width: 1800px;
        padding: 1rem 1.5rem 1rem 1.5rem;
    }

    h1, h2, h3, h4, h5, p, label {
        color: #183247 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #183247 !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stWidgetLabel"] p {
        color: #183247 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #52697D !important;
    }

    .hero {
        background: linear-gradient(115deg, #164E70, #13849A);
        padding: 18px 24px;
        border-radius: 15px;
        margin-bottom: 18px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        margin: 0 0 7px 0;
    }

    .hero p {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 14px;
    }

    .section-title {
        color: #183247 !important;
        font-size: 1.15rem;
        font-weight: 750;
        margin: 8px 0 12px 0;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #AABBCD !important;
        border-radius: 8px !important;
        min-height: 42px !important;
    }

    [data-baseweb="select"] *,
    [data-baseweb="select"] input,
    [data-baseweb="select"] [role="combobox"] {
        color: #183247 !important;
        -webkit-text-fill-color: #183247 !important;
        opacity: 1 !important;
    }

    [data-baseweb="select"] svg {
        fill: #183247 !important;
        color: #183247 !important;
    }

    /* Dropdown options */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #183247 !important;
        -webkit-text-fill-color: #183247 !important;
    }

    [role="option"]:hover {
        background-color: #E7F1FA !important;
    }

    /* Forecast cards */
    .forecast-card {
        background: #FFFFFF;
        border: 1px solid #D8E2EC;
        border-radius: 9px;
        padding: 9px 4px;
        min-height: 94px;
        text-align: center;
        overflow-wrap: anywhere;
    }

    .forecast-time {
        color: #183247 !important;
        font-size: 13px;
        font-weight: 750;
    }

    .forecast-date {
        color: #52697D !important;
        font-size: 10px;
        margin-top: 3px;
    }

    .forecast-value {
        color: #1769AA !important;
        font-size: 18px;
        font-weight: 800;
        margin-top: 6px;
    }

    .forecast-unit {
        color: #52697D !important;
        font-size: 10px;
    }

    /* Buttons */
    div.stButton > button,
    div.stDownloadButton > button {
        background: #1769AA !important;
        color: #FFFFFF !important;
        border: 1px solid #1769AA !important;
        border-radius: 9px !important;
        min-height: 42px;
        font-weight: 700;
    }

    div.stButton > button *,
    div.stDownloadButton > button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #125487 !important;
        border-color: #125487 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D8E2EC;
        border-radius: 11px;
        padding: 10px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #52697D !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #183247 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    hr {
        border-color: #D8E2EC !important;
    }

    .footer {
        color: #52697D !important;
        text-align: center;
        font-size: 12px;
        padding-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. FIND ORIGINAL SOFTS PROJECT
# ============================================================

APP_DIR = Path(__file__).resolve().parent
CANDIDATES = [APP_DIR, *APP_DIR.parents]

ROOT = next(
    (
        path for path in CANDIDATES
        if (path / "models" / "SOFTS.py").is_file()
        and (path / "data" / "scaler_stats.csv").is_file()
        and (path / "data" / "windows_96_to_24.npz").is_file()
    ),
    None,
)

if ROOT is None:
    st.error(
        "SOFTS project files could not be located. "
        "Please check the project directory."
    )
    st.stop()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 4. PATHS AND CONSTANTS
# ============================================================

DATA_DIR = ROOT / "data"
SCALER_PATH = DATA_DIR / "scaler_stats.csv"
DATASET_PATH = DATA_DIR / "windows_96_to_24.npz"

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / (
        "long_term_forecast_Hanoi_PM25_SOFTS_PM25_ftMS_"
        "sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_"
        "ebtimeF_dtTrue_test"
    )
    / "checkpoint.pth"
)

FEATURES = [
    "pm25",
    "pm10",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "rainfall",
]

FORECAST_HOURS = 24
REFERENCE_LEVEL = 50.0
TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌫️ Hanoi Air Intelligence</h1>
        <p>
            Understand air quality and explore the next 24 hours
            of PM2.5 forecasts in Hanoi.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Air Quality Forecast</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 6. AREA SELECTION
# ============================================================

AREAS = [
    "Hanoi — City-wide",
    "Ba Dinh",
    "Hoan Kiem",
    "Dong Da",
    "Cau Giay",
    "Hai Ba Trung",
    "Thanh Xuan",
    "Long Bien",
    "Nam Tu Liem",
    "Bac Tu Liem",
    "Ha Dong",
    "Tay Ho",
    "Hoang Mai",
    "Thanh Tri",
    "Gia Lam",
    "Dong Anh",
    "Soc Son",
    "Son Tay",
]

selected_area = st.selectbox(
    "Select Area",
    options=AREAS,
    index=0,
    key="selected_area",
)


# ============================================================
# 7. SCALER STATISTICS
# ============================================================

def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


@st.cache_data(show_spinner=False)
def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler statistics file not found: {SCALER_PATH}"
        )

    df = pd.read_csv(SCALER_PATH)

    required = {"feature", "mean", "scale"}

    if not required.issubset(df.columns):
        raise ValueError(
            "scaler_stats.csv must contain feature, mean and scale."
        )

    df["feature_norm"] = df["feature"].map(normalize_name)
    df = df.set_index("feature_norm")

    missing = [f for f in FEATURES if f not in df.index]

    if missing:
        raise ValueError(
            "Missing scaler statistics for: " + ", ".join(missing)
        )

    for feature in FEATURES:
        mean = float(df.loc[feature, "mean"])
        scale = float(df.loc[feature, "scale"])

        if not np.isfinite(mean) or not np.isfinite(scale):
            raise ValueError(
                f"Invalid scaler statistics for {feature}."
            )

        if scale <= 0:
            raise ValueError(
                f"Scaler scale must be positive for {feature}."
            )

    return df


# ============================================================
# 8. LOAD TEST WINDOWS
# ============================================================

@st.cache_data(show_spinner=False)
def load_test_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {DATASET_PATH}"
        )

    with np.load(DATASET_PATH) as data:
        if "X_test" not in data.files:
            raise ValueError(
                "The dataset does not contain X_test."
            )

        x_test = np.asarray(data["X_test"], dtype=np.float32)

    if x_test.ndim != 3 or x_test.shape[1:] != (96, 7):
        raise ValueError(
            f"Expected X_test shape (N, 96, 7), got {x_test.shape}."
        )

    if len(x_test) == 0:
        raise ValueError("The test dataset is empty.")

    if not np.isfinite(x_test).all():
        raise ValueError("X_test contains NaN or infinite values.")

    return x_test


# ============================================================
# 9. LOAD SOFTS CHECKPOINT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Trained checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    try:
        from models.SOFTS import Model
    except Exception as exc:
        raise ImportError(
            f"Could not import SOFTS: {exc}"
        ) from exc

    # Must match the configuration used to train the checkpoint.
    args = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=7,
        dec_in=1,
        c_out=1,
        d_model=128,
        d_core=128,
        d_ff=256,
        e_layers=2,
        d_layers=1,
        dropout=0.0,
        use_norm=1,
        output_attention=False,
        moving_avg=25,
        factor=1,
        distil=True,
        activation="gelu",
        embed="timeF",
        freq="h",
        num_class=5,
    )

    model = Model(args)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint state dictionary.")

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model


# ============================================================
# 10. GENERATE FORECAST FOR A RANDOM TEST WINDOW
# ============================================================

def run_forecast(sample_index):
    scaler_df = load_scaler()
    model = load_model()
    x_test = load_test_data()

    if not 0 <= sample_index < len(x_test):
        raise IndexError("Selected test-window index is out of range.")

    x_scaled = x_test[sample_index]

    means = np.array(
        [float(scaler_df.loc[f, "mean"]) for f in FEATURES],
        dtype=np.float32,
    )

    scales = np.array(
        [float(scaler_df.loc[f, "scale"]) for f in FEATURES],
        dtype=np.float32,
    )

    encoder_input = torch.from_numpy(x_scaled).unsqueeze(0)

    decoder_input = torch.zeros(
        (1, FORECAST_HOURS, 1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(
            encoder_input,
            None,
            decoder_input,
            None,
        )

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim != 3:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if output.shape[1] != FORECAST_HOURS:
        raise ValueError(
            "The model did not return 24 forecast steps. "
            f"Actual shape: {tuple(output.shape)}"
        )

    if output.shape[-1] == 7:
        output = output[:, :, 0:1]
    elif output.shape[-1] != 1:
        raise ValueError(
            "Expected one PM2.5 output channel or seven feature "
            f"channels. Actual shape: {tuple(output.shape)}"
        )

    prediction_scaled = (
        output.detach().cpu().numpy().reshape(FORECAST_HOURS)
    )

    forecast = prediction_scaled * scales[0] + means[0]
    history = x_scaled[:, 0] * scales[0] + means[0]

    return (
        np.maximum(history, 0.0).astype(float),
        np.maximum(forecast, 0.0).astype(float),
        len(x_test),
    )


# ============================================================
# 11. FORECAST CHART
# ============================================================

def build_chart(history, forecast):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    history_hours = np.arange(-95, 1)
    forecast_hours = np.arange(1, 25)

    ax.axvspan(-95, 0, color="#3B82F6", alpha=0.06)
    ax.axvspan(1, 24, color="#F59E0B", alpha=0.10)

    ax.plot(
        history_hours,
        history,
        color="#2864FF",
        linewidth=2,
        label="Historical PM2.5",
    )

    ax.plot(
        [0, 1],
        [history[-1], forecast[0]],
        color="#E58B13",
        linestyle="--",
        linewidth=1.5,
    )

    ax.plot(
        forecast_hours,
        forecast,
        color="#E58B13",
        marker="o",
        markersize=3,
        linewidth=2,
        label="SOFTS Forecast",
    )

    ax.axhline(
        REFERENCE_LEVEL,
        color="#64748B",
        linestyle="--",
        linewidth=1,
        label=f"Reference: {REFERENCE_LEVEL:.0f} µg/m³",
    )

    ax.axvline(0, color="#64748B", linestyle=":", linewidth=1)

    peak_index = int(np.argmax(forecast))
    peak_value = float(forecast[peak_index])

    ax.scatter(
        [peak_index + 1],
        [peak_value],
        color="#E58B13",
        edgecolors="white",
        s=45,
        zorder=5,
    )

    ax.set_xlim(-95, 24)
    ax.set_xlabel("Relative Time (Hours)", color="#183247")
    ax.set_ylabel("PM2.5 (µg/m³)", color="#183247")

    ax.set_xticks([-95, -72, -48, -24, 0, 6, 12, 18, 24])
    ax.set_xticklabels(
        [
            "-95h", "-72h", "-48h", "-24h", "Now",
            "+6h", "+12h", "+18h", "+24h",
        ]
    )

    ax.tick_params(colors="#475569")
    ax.grid(axis="y", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


# ============================================================
# 12. SESSION STATE
# ============================================================

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "forecast_error" not in st.session_state:
    st.session_state.forecast_error = None


# ============================================================
# 13. GENERATE FORECAST BUTTON
# ============================================================

if st.button(
    "▶ Generate Forecast",
    type="primary",
    use_container_width=True,
):
    st.session_state.forecast_result = None
    st.session_state.forecast_error = None

    try:
        with st.spinner(
            "Selecting a random test window and generating forecast..."
        ):
            total_windows = len(load_test_data())

            random_index = int(
                np.random.default_rng().integers(0, total_windows)
            )

            history, forecast, _ = run_forecast(random_index)

            st.session_state.forecast_result = {
                "history": history,
                "forecast": forecast,
                "sample_index": random_index,
                "area": selected_area,
            }

    except Exception as exc:
        st.session_state.forecast_error = str(exc)


# ============================================================
# 14. DISPLAY ERRORS
# ============================================================

if st.session_state.forecast_error:
    st.error(
        "The forecast could not be generated. "
        "Please check the model configuration and checkpoint."
    )

    with st.expander("Technical Details", expanded=True):
        st.code(st.session_state.forecast_error, language="text")


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

result = st.session_state.forecast_result

if result is None:
    if not st.session_state.forecast_error:
        st.info(
            "Select an area and click Generate Forecast to view "
            "the forecast for a randomly selected test window."
        )

else:
    history = result["history"]
    forecast = result["forecast"]
    sample_index = result["sample_index"]
    area = result["area"]

    average_value = float(np.mean(forecast))
    peak_index = int(np.argmax(forecast))
    peak_value = float(forecast[peak_index])
    end_value = float(forecast[-1])

    hours_above = int(np.sum(forecast > REFERENCE_LEVEL))

    # Summary
    st.markdown(f"### Forecast Summary — {area}")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Average PM2.5",
        f"{average_value:.1f} µg/m³",
    )

    metric2.metric(
        "Peak PM2.5",
        f"{peak_value:.1f} µg/m³",
        f"Forecast step {peak_index + 1}",
    )

    metric3.metric(
        "After 24 Hours",
        f"{end_value:.1f} µg/m³",
    )

    metric4.metric(
        "Hours Above Reference",
        f"{hours_above}/24",
    )

    # Display clock times in Hanoi timezone.
    # These times are not linked to timestamps in X_test.
    now_hanoi = datetime.now(TIMEZONE)

    first_time = (
        now_hanoi.replace(minute=0, second=0, microsecond=0)
        + timedelta(hours=1)
    )

    forecast_times = [
        first_time + timedelta(hours=i)
        for i in range(FORECAST_HOURS)
    ]

    report = pd.DataFrame(
        {
            "Date": [t.strftime("%d/%m/%Y") for t in forecast_times],
            "Time": [t.strftime("%H:%M") for t in forecast_times],
            "Predicted PM2.5 (µg/m³)": np.round(forecast, 2),
        }
    )

    # Side-by-side layout: chart left, all 24 hours right.
    chart_col, forecast_col = st.columns(
        [1.35, 1],
        gap="medium",
    )

    with chart_col:
        st.markdown("### PM2.5 Forecast Trend")

        fig = build_chart(history, forecast)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with forecast_col:
        st.markdown("### Next 24 Hours")
        st.caption("Hanoi local time (UTC+7)")

        # Four rows of six cards: 24 hours total.
        for row_start in range(0, 24, 6):
            columns = st.columns(6, gap="small")

            for offset, column in enumerate(columns):
                index = row_start + offset
                timestamp = forecast_times[index]
                value = float(forecast[index])

                with column:
                    st.markdown(
                        f"""
                        <div class="forecast-card">
                            <div class="forecast-time">
                                {timestamp.strftime("%H:%M")}
                            </div>
                            <div class="forecast-date">
                                {timestamp.strftime("%d/%m")}
                            </div>
                            <div class="forecast-value">
                                {value:.1f}
                            </div>
                            <div class="forecast-unit">
                                µg/m³
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.download_button(
            "Download Forecast CSV",
            data=report.to_csv(index=False).encode("utf-8-sig"),
            file_name="hanoi_pm25_forecast_24h.csv",
            mime="text/csv",
            use_container_width=True,
        )

    

    # Health recommendations
    st.markdown("### Health Recommendations")

    if average_value <= 15:
        recommendation = (
            "The predicted PM2.5 level is relatively low. "
            "Continue normal activities and monitor local conditions."
        )
    elif average_value <= 35:
        recommendation = (
            "Sensitive individuals may wish to reduce prolonged "
            "outdoor exertion if they experience discomfort."
        )
    elif average_value <= 55:
        recommendation = (
            "Consider reducing prolonged outdoor exertion, "
            "especially if you are sensitive to air pollution."
        )
    elif average_value <= 150:
        recommendation = (
            "Consider limiting outdoor exertion. Sensitive groups "
            "should take additional precautions."
        )
    else:
        recommendation = (
            "Reduce exposure to outdoor pollution and follow "
            "current local public-health guidance."
        )

    st.info(recommendation)

    st.caption(
        "The 50 µg/m³ reference line is illustrative and is not "
        "a universal official air-quality threshold."
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Hanoi Air Intelligence · SOFTS Forecasting System
    </div>
    """,
    unsafe_allow_html=True,
)
