# One-Station Weather Forecast (LSTM, Streamlit)

Forecast next-day **maximum air temperature (`atmp_max`)** from the last 7 days of features using a single weather station.
Includes a Streamlit app for instant deployment, a training script, and a Dockerfile for containerized runs.

## Project Structure

```
one_station_forecast/
├── app.py                     # Streamlit UI
├── src/
│   ├── train.py              # Train LSTM and save artifacts
│   ├── model.py              # PyTorch model + load/save helpers
│   └── data_utils.py         # Loading, cleaning, windowing
├── data/
│   └── good_mawn_records_aetna.csv   # Sample station dataset
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── .gitignore
```

## Local Quickstart (venv)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Docker (Containerized Run)
```bash
docker build -t one-station-forecast .
docker run --rm -p 8501:8501 one-station-forecast
# open http://localhost:8501
```

Mount custom data at runtime:
```bash
docker run --rm -p 8501:8501 -v $PWD/data:/app/data one-station-forecast
```

If `torch` fails during build, the Dockerfile includes a CPU-only fallback step.

## Hugging Face Spaces (Streamlit)
- Create a Streamlit Space, push this repo, leave `requirements.txt` as-is.
- Space auto-builds and serves the app.
