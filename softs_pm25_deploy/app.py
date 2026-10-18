
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0e1117;
    color: #f1f5f9;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2rem 2.5rem 2rem;
}

h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

h2, h3 {
    font-weight: 700 !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.section-title {
    display: flex;
    align-items: center;
    font-size: 1.2rem;
    font-weight: 750;
    margin-bottom: 24px;
    color: #f8fafc;
}

.step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    min-width: 32px;
    background: #2864ed;
    color: #ffffff;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    margin-right: 12px;
}

.info-card {
    background: #171c26;
    border: 1px solid #283244;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.info-title {
    color: #a6b4cb;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 8px;
}

.info-value {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 750;
}

.info-description {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 5px;
}

div.stButton > button {
    width: 100%;
    min-height: 52px;
    background: #2864ed;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    transition: background 0.2s ease;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: #ffffff;
    border: none;
}

div[data-testid="stMetric"] {
    background: #171c26;
    padding: 17px;
    border: 1px solid #283244;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #a6b4cb;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #283244;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #283244;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

.app-footer {
    color: #64748b;
    font-size: 0.8rem;
    text-align: center;
    padding-top: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOCATE THE SOFTS PROJECT
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
        "The application could not locate the SOFTS project files. "
        "Please verify the project directory structure."
    )
    st.stop()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
SCALER_PATH = DATA_DIR / "scaler_stats.csv"
DATASET_PATH = DATA_DIR / "windows_96_to_24.npz"

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / (
        "long_term_forecast_Hanoi_PM25_SOFTS_PM25_ftMS_"
        "sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_"
        "ebtimeF_dtTrue_test"
    )
)

CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pth"

FEATURES = [
    "pm25",
    "pm10",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "rainfall",
]


# ============================================================
# 4. LOAD SCALER STATISTICS
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
            "The scaler statistics file could not be found."
        )

    df = pd.read_csv(SCALER_PATH)

    required_columns = {"feature", "mean", "scale"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The scaler file must contain feature, mean, and scale columns."
        )

    df["feature_norm"] = df["feature"].map(normalize_name)
    df = df.set_index("feature_norm")

    missing = [
        feature for feature in FEATURES
        if feature not in df.index
    ]

    if missing:
        raise ValueError(
            f"Missing scaler statistics for: {', '.join(missing)}"
        )

    return df


# ============================================================
# 5. LOAD THE TRAINED SOFTS MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "The trained model checkpoint could not be found. "
            "Please verify the checkpoint directory."
        )

    try:
        from models.SOFTS import Model
    except Exception as exc:
        raise ImportError(
            f"Unable to import the SOFTS model: {exc}"
        ) from exc

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

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ============================================================
# 6. LOAD INTEGRATED HISTORICAL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_test_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "The integrated forecasting dataset could not be found."
        )

    with np.load(DATASET_PATH) as dataset:
        if "X_test" not in dataset.files:
            raise ValueError(
                "The dataset does not contain the required input data."
            )

        x_test = np.asarray(
            dataset["X_test"],
            dtype=np.float32,
        )

    if x_test.ndim != 3 or x_test.shape[1:] != (96, 7):
        raise ValueError(
            "The input data must have 96 time steps and 7 features."
        )

    if len(x_test) == 0:
        raise ValueError("The integrated dataset is empty.")

    return x_test


# ============================================================
# 7. GENERATE PM2.5 FORECAST
# ============================================================

def run_forecast(sample_index=0):
    scaler_df = load_scaler()
    model = load_model()
    x_test = load_test_data()

    sample_index = min(
        max(int(sample_index), 0),
        len(x_test) - 1,
    )

    x_scaled = x_test[sample_index]

    means = np.array(
        [
            float(scaler_df.loc[feature, "mean"])
            for feature in FEATURES
        ],
        dtype=np.float32,
    )

    scales = np.array(
        [
            float(scaler_df.loc[feature, "scale"])
            for feature in FEATURES
        ],
        dtype=np.float32,
    )

    if np.any(scales <= 0):
        raise ValueError(
            "Invalid scaling statistics. Scale values must be positive."
        )

    encoder_input = torch.from_numpy(
        x_scaled
    ).unsqueeze(0)

    decoder_input = torch.zeros(
        (1, 24, 1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(
            encoder_input,
            None,
            decoder_input,
            None,
        )

        if isinstance(output, tuple):
            output = output[0]

    if output.ndim != 3 or output.shape[1] != 24:
        raise ValueError(
            "The model returned an unexpected forecast shape."
        )

    if output.shape[-1] == 7:
        output = output[:, :, 0:1]
    elif output.shape[-1] != 1:
        raise ValueError(
            "The model output does not contain a valid PM2.5 channel."
        )

    prediction_scaled = (
        output.detach().cpu().numpy().reshape(24)
    )

    # Convert standardized PM2.5 values back to original units.
    forecast = (
        prediction_scaled * scales[0] + means[0]
    )

    history = (
        x_scaled[:, 0] * scales[0] + means[0]
    )

    forecast = np.maximum(forecast, 0.0)
    history = np.maximum(history, 0.0)

    return (
        history.astype(float),
        forecast.astype(float),
    )


# ============================================================
# 8. FORECAST VISUALIZATION
# ============================================================

def build_chart(history, forecast):
    fig, ax = plt.subplots(
        figsize=(11, 4.6),
        dpi=130,
    )

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    historical_hours = np.arange(-95, 1)
    forecast_hours = np.arange(1, 25)

    ax.axvspan(
        -95, 0,
        color="#3b82f6",
        alpha=0.05,
    )

    ax.axvspan(
        1, 24,
        color="#f59e0b",
        alpha=0.10,
    )

    ax.plot(
        historical_hours,
        history,
        color="#2864ff",
        linewidth=2,
        label="Historical PM2.5",
    )

    ax.plot(
        [0, 1],
        [history[-1], forecast[0]],
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
    )

    ax.plot(
        forecast_hours,
        forecast,
        color="#f59e0b",
        marker="o",
        markersize=3,
        linewidth=2,
        label="SOFTS Forecast",
    )

    ax.axhline(
        50,
        color="#64748b",
        linestyle="--",
        linewidth=1,
        label="Reference Level: 50 µg/m³",
    )

    ax.axvline(
        0,
        color="#475569",
        linestyle=":",
        linewidth=1.2,
    )

    peak_index = int(np.argmax(forecast))
    peak_value = float(forecast[peak_index])

    ax.scatter(
        [peak_index + 1],
        [peak_value],
        color="#f59e0b",
        edgecolors="white",
        linewidths=1,
        s=50,
        zorder=5,
    )

    ax.annotate(
        f"{peak_value:.1f} µg/m³",
        (peak_index + 1, peak_value),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )

    ax.set_xlim(-95, 24)

    ax.set_xlabel("Relative Time (Hours)")
    ax.set_ylabel("PM2.5 (µg/m³)")

    ax.set_xticks(
        [-95, -72, -48, -24, 0, 6, 12, 18, 24]
    )

    ax.set_xticklabels(
        [
            "-95h", "-72h", "-48h", "-24h",
            "Now", "+6h", "+12h", "+18h", "+24h",
        ]
    )

    ax.grid(axis="y", alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    return fig


# ============================================================
# 9. APPLICATION HEADER
# ============================================================

st.title("Air Quality Forecast")

st.caption(
    "PM2.5 Forecasting · Powered by SOFTS"
)

st.divider()


# ============================================================
# 10. APPLICATION STATE
# ============================================================

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None


# ============================================================
# 11. MAIN DASHBOARD
# ============================================================

left, right = st.columns(
    [0.8, 2.0],
    gap="large",
)


# ------------------------------------------------------------
# LEFT PANEL: CONTROLS AND KEY METRICS
# ------------------------------------------------------------

with left:
    st.markdown(
        '<div class="section-title">'
        '<span class="step">1</span>Run Forecast</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Forecast Configuration")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Input Variables</div>
            <div class="info-value">7 Features</div>
            <div class="info-description">
                Multivariate environmental data
            </div>
        </div>

        <div class="info-card">
            <div class="info-title">Historical Window</div>
            <div class="info-value">96 Hours</div>
            <div class="info-description">
                Historical observations
            </div>
        </div>

        <div class="info-card">
            <div class="info-title">Forecast Horizon</div>
            <div class="info-value">24 Hours</div>
            <div class="info-description">
                Future PM2.5 concentrations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_clicked = st.button(
        "▶  RUN FORECAST",
        use_container_width=True,
    )

    if run_clicked:
        try:
            with st.spinner(
                "Generating PM2.5 forecast..."
            ):
                history, forecast = run_forecast(
                    sample_index=0
                )

                st.session_state.forecast_result = {
                    "history": history,
                    "forecast": forecast,
                }

        except Exception as exc:
            st.session_state.forecast_result = None

            st.error(
                f"Forecast generation failed: {exc}"
            )

    result = st.session_state.forecast_result

    if result is not None:
        history = result["history"]
        forecast = result["forecast"]

        peak_index = int(np.argmax(forecast))
        peak_value = float(forecast[peak_index])
        exceed_hours = int(np.sum(forecast > 50))

        st.divider()

        st.markdown("### Forecast Summary")

        st.metric(
            "Peak PM2.5",
            f"{peak_value:.1f} µg/m³",
            f"Hour +{peak_index + 1}",
        )

        st.metric(
            "PM2.5 After 24 Hours",
            f"{forecast[-1]:.1f} µg/m³",
        )

        st.metric(
            "Hours Above 50 µg/m³",
            f"{exceed_hours} / 24",
        )


# ------------------------------------------------------------
# RIGHT PANEL: CHART AND FORECAST DETAILS
# ------------------------------------------------------------

with right:
    st.markdown(
        '<div class="section-title">'
        '<span class="step">2</span>Forecast Results</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.forecast_result

    if result is None:
        st.info(
            "Run the forecast to view the PM2.5 trend "
            "and the next 24 hours of predictions."
        )

    else:
        history = result["history"]
        forecast = result["forecast"]

        # Main chart
        st.markdown("### PM2.5 Forecast Trend")

        fig = build_chart(
            history,
            forecast,
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

        # Forecast details
        st.markdown("### 24-Hour Forecast Details")

        report = pd.DataFrame({
            "Forecast Time": [
                f"+{hour}h"
                for hour in range(1, 25)
            ],
            "Predicted PM2.5 (µg/m³)": np.round(
                forecast,
                2,
            ),
        })

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download Forecast Report",
            data=report.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="pm25_forecast_24h.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        Air Quality Forecast · SOFTS Forecasting System
    </div>
    """,
    unsafe_allow_html=True,
)
