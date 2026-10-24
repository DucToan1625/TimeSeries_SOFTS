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
