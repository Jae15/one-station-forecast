import os, pickle, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from src.model import LSTMForecaster
from src.data_utils import load_station_csv, scale_features, make_windows, FEATURES, TARGET

st.set_page_config(page_title="One-Station Forecast", layout="wide")
st.title("🌤️ One-Station Weather Forecast (LSTM)")
st.caption("Forecast next-day `atmp_max` from the last 7 days.")

data_path = "data/good_mawn_records_aetna.csv"
if not os.path.exists(data_path):
    st.error("Expected data file not found: data/good_mawn_records_aetna.csv")
    st.stop()

df = load_station_csv(data_path)
st.subheader("Recent data")
st.dataframe(df.tail(10))

# Ensure artifacts
if not (os.path.exists("model.pth") and os.path.exists("scaler.pkl")):
    st.info("Artifacts missing. Training now...")
    from src.train import train
    train(data_path=data_path, seq_len=7, epochs=30, batch_size=32, lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecaster(len(FEATURES)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df_scaled, _ = scale_features(df)
X, y = make_windows(df_scaled, seq_len=7)
if len(X) == 0:
    st.error("Not enough rows to create a 7-day window. Add more data.")
else:
    with torch.no_grad():
        yhat = model(torch.tensor(X[-1:]).to(device)).cpu().numpy()[0]
    st.subheader("Prediction (scaled units)")
    st.write(f"Next-day `{TARGET}` (scaled 0..1): **{yhat:.3f}**")

    # Plot
    fig = plt.figure()
    N = min(100, len(y))
    plt.plot(np.arange(N), y[-N:], label="True (scaled)")
    plt.plot([N], [yhat], "o", label="Forecast")
    plt.xlabel("Time steps (days)"); plt.ylabel(f"{TARGET} (scaled)"); plt.legend()
    st.pyplot(fig)

st.info("Next iteration: persist target scaler and inverse-transform for real units; add metrics and date-based split.")
