from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

FEATURES = ["atmp_min","atmp_max","relh_min","relh_max","pcpn","wspd_max","srad"]
TARGET = "atmp_max"

def load_station_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    df = df.dropna(subset=[c for c in FEATURES+[TARGET] if c in df.columns])
    return df[["date"] + [c for c in FEATURES if c in df.columns] + [TARGET]]

def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(df[FEATURES])
    sdf = pd.DataFrame(Xs, columns=FEATURES, index=df.index)
    sdf["date"] = df["date"].values
    sdf[TARGET] = df[TARGET].values
    return sdf, scaler

def make_windows(df_scaled: pd.DataFrame, seq_len: int = 7):
    X, y = [], []
    vals = df_scaled[FEATURES + [TARGET]].values
    for i in range(len(vals) - seq_len):
        X.append(vals[i:i+seq_len, :len(FEATURES)])
        y.append(vals[i+seq_len, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
