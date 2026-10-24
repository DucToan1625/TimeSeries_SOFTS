import sys
import io
import random
import html
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
INPUT_HOURS = 96
FORECAST_HOURS = 24
NUM_FEATURES = 7
REFERENCE_LEVEL = 50.0

FEATURE_COLUMNS = [
    "pm25", "pm10", "temperature", "humidity",
    "wind_speed", "pressure", "rainfall",
]
TARGET_COLUMN = "pm25"

AREAS = [
    "Hanoi — City-wide", "Ba Dinh", "Hoan Kiem", "Dong Da",
    "Cau Giay", "Hai Ba Trung", "Thanh Xuan", "Long Bien",
    "Nam Tu Liem", "Bac Tu Liem", "Ha Dong", "Tay Ho",
    "Hoang Mai", "Thanh Tri", "Gia Lam", "Dong Anh",
    "Soc Son", "Son Tay",
]

LANGUAGES = {
    "English": {
        "settings": "Forecast Settings",
        "model": "SOFTS PM2.5 Model",
        "model_caption": "96-hour history · 24-hour forecast",
        "area": "Select area",
        "horizon": "Forecast horizon: 24 hours",
        "history": "Input history: 96 hours",
        "timezone": "Time zone: Vietnam · UTC+7",
        "run": "▶  Run Forecast",
        "performance": "Model Performance",
        "mae_note": "Compared with ground truth when available",
        "mse_note": "Mean squared error when ground truth is available",
        "rate_note": "Display indicator, not verified model accuracy",
        "files": "✓ Project files detected",
        "chart": "📈  Forecast Chart",
        "forecast": "🕒  24-Hour Forecast",
        "table": "▦  Data Table",
        "peak": "Peak at",
        "average": "Forecast average",
        "reference": "reference",
        "above": "Above the 50 µg/m³ reference for about",
        "hours": "hours",
        "chart_title": "Historical PM2.5 and 24-Hour Forecast",
        "xlabel": "Hanoi local time (UTC+7)",
        "ylabel": "PM2.5 (µg/m³)",
        "history_label": "Historical PM2.5",
        "forecast_label": "SOFTS Forecast",
        "explanation": "AI Explanation",
        "forecast_heading": "PM2.5 Forecast · Next 24 Hours",
        "forecast_caption": "Forecast values for the next 24 hours in Vietnam local time (UTC+7).",
        "data_heading": "Hourly Forecast Data",
        "data_caption": "24 hourly forecast timestamps in Hanoi local time (UTC+7).",
        "date": "Date",
        "time": "Time",
        "predicted": "Predicted PM2.5 (µg/m³)",
        "download_csv": "⬇  Download Forecast CSV",
        "download_png": "⬇  Download Chart PNG",
        "good": "Good",
        "moderate": "Moderate",
        "high": "High",
        "poor": "Poor",
        "error": "Forecast failed",
        "loading": "Generating SOFTS forecast...",
        "init_error": "Unable to initialize the forecasting system.",
        "check_paths": "Check the project directory, scaler_stats.csv, windows_96_to_24.npz and SOFTS checkpoint.",
        "footer": "Hanoi Air Intelligence · SOFTS PM2.5 Forecasting · Research Prototype",
        "above_text": "The predicted average PM2.5 exceeds the 50 µg/m³ reference level. Monitor local air quality.",
        "below_text": "The predicted average PM2.5 is below the 50 µg/m³ reference level, although individual hours may be higher.",
        "generated": "Generated at",
        "no_ground_truth": "N/A",
        "units": "µg/m³",
    },
    "Tiếng Việt": {
        "settings": "Cấu hình dự báo",
        "model": "Mô hình SOFTS PM2.5",
        "model_caption": "Dữ liệu lịch sử 96 giờ · Dự báo 24 giờ",
        "area": "Chọn khu vực",
        "horizon": "Khoảng dự báo: 24 giờ",
        "history": "Dữ liệu đầu vào: 96 giờ",
        "timezone": "Múi giờ: Việt Nam · UTC+7",
        "run": "▶  Chạy dự báo",
        "performance": "Hiệu năng mô hình",
        "mae_note": "So sánh với dữ liệu thực tế nếu có",
        "mse_note": "Sai số bình phương trung bình nếu có dữ liệu thực tế",
        "rate_note": "Chỉ số hiển thị, không phải độ chính xác đã xác minh",
        "files": "✓ Đã tìm thấy tệp dự án",
        "chart": "📈  Biểu đồ dự báo",
        "forecast": "🕒  Dự báo 24 giờ",
        "table": "▦  Bảng dữ liệu",
        "peak": "Cao nhất lúc",
        "average": "PM2.5 trung bình dự báo",
        "reference": "ngưỡng tham chiếu",
        "above": "Số giờ vượt ngưỡng tham chiếu 50 µg/m³ khoảng",
        "hours": "giờ",
        "chart_title": "PM2.5 lịch sử và dự báo trong 24 giờ",
        "xlabel": "Thời gian tại Hà Nội (UTC+7)",
        "ylabel": "PM2.5 (µg/m³)",
        "history_label": "PM2.5 lịch sử",
        "forecast_label": "Dự báo SOFTS",
        "explanation": "Giải thích của AI",
        "forecast_heading": "Dự báo PM2.5 · 24 giờ tiếp theo",
        "forecast_caption": "Giá trị dự báo theo giờ tại Việt Nam (UTC+7).",
        "data_heading": "Dữ liệu dự báo theo giờ",
        "data_caption": "24 mốc thời gian dự báo theo giờ tại Hà Nội (UTC+7).",
        "date": "Ngày",
        "time": "Giờ",
        "predicted": "PM2.5 dự báo (µg/m³)",
        "download_csv": "⬇  Tải dữ liệu dự báo CSV",
        "download_png": "⬇  Tải biểu đồ PNG",
        "good": "Tốt",
        "moderate": "Trung bình",
        "high": "Cao",
        "poor": "Kém",
        "error": "Dự báo thất bại",
        "loading": "Đang tạo dự báo SOFTS...",
        "init_error": "Không thể khởi tạo hệ thống dự báo.",
        "check_paths": "Kiểm tra thư mục dự án, scaler_stats.csv, windows_96_to_24.npz và checkpoint SOFTS.",
        "footer": "Hanoi Air Intelligence · Dự báo PM2.5 bằng SOFTS · Bản thử nghiệm nghiên cứu",
        "above_text": "PM2.5 trung bình dự báo vượt ngưỡng tham chiếu 50 µg/m³. Hãy theo dõi chất lượng không khí.",
        "below_text": "PM2.5 trung bình dự báo dưới ngưỡng tham chiếu 50 µg/m³, tuy nhiên một số giờ riêng lẻ có thể cao hơn.",
        "generated": "Thời điểm tạo dự báo",
        "no_ground_truth": "Không có",
        "units": "µg/m³",
    },
}

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

MODEL_FILE = MODEL_DIR / "SOFTS.py"
SCALER_FILE = DATA_DIR / "scaler_stats.csv"
WINDOW_FILE = DATA_DIR / "windows_96_to_24.npz"

CHECKPOINT_FILE = (
    CHECKPOINT_DIR
    / "long_term_forecast_Hanoi_PM25_SOFTS_PM25_ftMS_sl96_ll48_pl24_dm128_el2_dl1_df256_fc1_ebtimeF_dtTrue_test"
    / "checkpoint.pth"
)

# ============================================================
# 3. CSS — STREAMLIT CLOUD LAYOUT FIX
# ============================================================

st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       GLOBAL PAGE
    -------------------------------------------------------- */
    html, body, [data-testid="stAppViewContainer"] {
        background: #f5f7fb !important;
        color: #172033 !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
    }

    .main .block-container {
        max-width: 1600px !important;
        width: 100% !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
        overflow: visible !important;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.96) !important;
        z-index: 100 !important;
    }

    /* Hide Streamlit Deploy button where supported */
    [data-testid="stAppDeployButton"],
    [data-testid="stDeployButton"],
    .stDeployButton,
    button[kind="header"],
    [data-testid="stToolbar"] button[title*="Deploy"],
    [data-testid="stToolbar"] button[aria-label*="Deploy"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Keep toolbar from covering the application content */
    [data-testid="stToolbar"] {
        right: 0.5rem !important;
        top: 0.25rem !important;
        z-index: 101 !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* --------------------------------------------------------
       HEADINGS AND TEXT
    -------------------------------------------------------- */
    h1, h2, h3, h4, h5, h6,
    p, label, span, small, li {
        overflow-wrap: anywhere;
    }

    h1, h2, h3 {
        color: #172554 !important;
        letter-spacing: -0.025em;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #334155;
    }

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */
    .hero {
        background: linear-gradient(120deg, #ffffff 0%, #edf4ff 100%);
        border: 1px solid #dbe5f3;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.045);
        overflow-wrap: anywhere;
    }

    .hero-title {
        color: #172554;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        line-height: 1.2;
        font-weight: 850;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }

    /* --------------------------------------------------------
       CARDS AND METRICS
    -------------------------------------------------------- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        min-width: 0 !important;
        overflow-wrap: anywhere;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 800 !important;
        overflow-wrap: anywhere !important;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 42px !important;
        height: auto !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button[kind="primary"] {
        background: #4338ca !important;
        color: #ffffff !important;
        border: 1px solid #4338ca !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #3730a3 !important;
        border-color: #3730a3 !important;
    }

    .stDownloadButton > button {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* --------------------------------------------------------
       TABS — HIGH CONTRAST, NO OVERFLOW
    -------------------------------------------------------- */
    div[data-testid="stTabs"] {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        background: #edf2fb !important;
        border: 1px solid #d5deed !important;
        border-radius: 14px !important;
        padding: 6px !important;
        margin-top: 12px !important;
        overflow: visible !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        width: 100% !important;
        min-width: 0 !important;
        white-space: normal !important;
        background: transparent !important;
        border-bottom: 0 !important;
        overflow: visible !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        flex: 1 1 140px !important;
        min-width: 0 !important;
        min-height: 42px !important;
        padding: 9px 14px !important;
        border-radius: 9px !important;
        border: 1px solid transparent !important;
        background: #e2e8f0 !important;
        color: #1e293b !important;
        opacity: 1 !important;
        white-space: normal !important;
        font-size: 0.92rem !important;
        font-weight: 750 !important;
        line-height: 1.35 !important;
        text-align: center !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] *,
    div[data-testid="stTabs"] button[data-baseweb="tab"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"] span {
        color: #1e293b !important;
        opacity: 1 !important;
        font-weight: 750 !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background: #dbeafe !important;
        border-color: #93c5fd !important;
        color: #1e3a8a !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        border: 1px solid #a5b4fc !important;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.12) !important;
        color: #3730a3 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] span {
        color: #3730a3 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* IMPORTANT: do not constrain tab panel height.
       Let the page expand naturally with all forecast content. */
    div[data-testid="stTabs"] [data-baseweb="tab-panel"],
    div[data-testid="stTabContent"],
    div[data-testid="stVerticalBlock"] {
        min-width: 0 !important;
        max-width: 100% !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        padding-top: 1rem !important;
    }

    /* --------------------------------------------------------
       DATAFRAME AND CHART
    -------------------------------------------------------- */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        max-width: 100% !important;
    }

    /* --------------------------------------------------------
       FORECAST CARDS
    -------------------------------------------------------- */
    .forecast-card {
        min-width: 0;
        width: 100%;
        min-height: 88px;
        box-sizing: border-box;
        padding: 9px 5px;
        border: 1px solid #dbe5f3;
        border-radius: 12px;
        background: #ffffff;
        text-align: center;
        overflow-wrap: anywhere;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.035);
    }

    .forecast-card .forecast-time {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .forecast-card .forecast-value {
        color: #172554;
        font-size: clamp(0.95rem, 1.5vw, 1.25rem);
        font-weight: 850;
        line-height: 1.25;
    }

    .forecast-card .forecast-unit {
        color: #64748b;
        font-size: 0.7rem;
        margin-top: 3px;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */
    .app-footer {
        margin-top: 26px;
        padding: 16px 8px 4px;
        border-top: 1px solid #dbe3ef;
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* --------------------------------------------------------
       RESPONSIVE LAYOUT
    -------------------------------------------------------- */
    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero {
            padding: 18px;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            flex-basis: 120px !important;
        }
    }

    @media (max-width: 600px) {
        .hero-title {
            font-size: 1.55rem;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            flex-basis: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
