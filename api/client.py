"""Small client helper for Streamlit or other Python callers."""
import os
import requests


def forecast_via_api(values, base_url=None, api_key=None, timeout=30):
    base_url = (base_url or os.getenv("SOFTS_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    api_key = api_key if api_key is not None else os.getenv("API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{base_url}/forecast", json={"values": values}, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    return response.json()["forecast_pm25"]
