import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B,T,H)
        last = out[:, -1, :]       # (B,H)
        y = self.fc(last)          # (B,1)
        return y.squeeze(-1)       # (B,)
