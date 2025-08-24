from __future__ import annotations
import argparse, pickle, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from model import LSTMForecaster
from data_utils import load_station_csv, scale_features, make_windows, FEATURES, TARGET

def train(data_path="data/good_mawn_records_aetna.csv", seq_len=7, epochs=50, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_station_csv(data_path)
    df_scaled, scaler = scale_features(df)
    X, y = make_windows(df_scaled, seq_len)
    if len(X) < 10:
        print("[WARN] Very small dataset after windowing.")
    split = int(0.8 * len(X)) if len(X) > 0 else 0
    Xtr, Xv = X[:split], X[split:]
    ytr, yv = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=batch_size, shuffle=True) if split>0 else None
    val_loader = DataLoader(TensorDataset(torch.tensor(Xv), torch.tensor(yv)), batch_size=batch_size, shuffle=False) if len(X)-split>0 else None

    model = LSTMForecaster(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best = float("inf")
    for ep in range(1, epochs+1):
        if train_loader is not None:
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
        # validation
        if val_loader is not None:
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    pv = model(xb.to(device))
                    vp.append(pv.cpu().numpy()); vt.append(yb.numpy())
            vp, vt = np.concatenate(vp) if vp else np.array([]), np.concatenate(vt) if vt else np.array([])
            if vp.size:
                mae = mean_absolute_error(vt, vp)
                r2 = r2_score(vt, vp) if len(set(vt))>1 else float('nan')
                if mae < best:
                    best = mae
                    torch.save(model.state_dict(), "model.pth")
                    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
        print(f"Epoch {ep:03d} | best_val_MAE={best if best!=float('inf') else float('nan'):.4f}")
    # Ensure artifacts saved at least once
    if not (os.path.exists("model.pth")):
        torch.save(model.state_dict(), "model.pth")
        with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    print("Training complete. Saved model.pth and scaler.pkl.")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/good_mawn_records_aetna.csv")
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.seq_len, args.epochs, args.batch_size, args.lr)
