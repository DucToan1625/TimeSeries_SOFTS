# SOFTS PM2.5 Forecast Dashboard

Streamlit dashboard based on the supplied UI design. It shows 96 hours of history and the next 24-hour PM2.5 forecast.

## Important folder layout

Place `app.py` in the **root of the SOFTS repository**, alongside `models/`, `data/`, and `checkpoints/`:

```text
SOFTS/
├── app.py
├── requirements.txt
├── models/
│   └── SOFTS.py
├── data/
│   ├── scaler_stats.csv
│   └── ...
└── checkpoints/
    └── long_term_forecast_Hanoi_PM25_SOFTS_PM25_.../
        └── checkpoint.pth
```

Do not publish private data or a private checkpoint to a public repository unless you are comfortable making them public. For a public deployment, use a private repository if your hosting provider supports it, or configure protected storage/secrets.

## Run locally

From the SOFTS repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The CSV uploaded to the app must contain at least 96 valid hourly rows and all seven feature columns:
`pm25, pm10, temperature, humidity, wind_speed, pressure, rainfall`.
Some common column aliases are supported. The app uses the last 96 valid rows.

## Deploy

### Streamlit Community Cloud
1. Push the SOFTS repository and this `app.py` to a GitHub repository.
2. Ensure `requirements.txt` is in the repository root.
3. Open https://share.streamlit.io/ and sign in.
4. Select **Create app**, choose your repository, branch, and `app.py`.
5. Deploy, then test with a CSV containing the seven variables.

### Notes
- This generated app is a deploy-ready starting point; it has not been deployed from this chat, so no live app URL exists yet.
- The checkpoint must be present at the path configured near the top of `app.py`.
- The app loads the checkpoint and runs the SOFTS model locally on the hosting machine.
- PyTorch CPU deployment may require a suitable build/runtime; check the host's resource limits for the checkpoint size and dependency compatibility.
- The displayed 50 µg/m³ line is a configurable reference line, not a universal statement about all regulatory standards.
- Forecast quality metrics such as MAE/RMSE require actual future observations. The app does not invent these metrics for future forecasts.
