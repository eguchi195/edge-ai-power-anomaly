import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# ── 設定 ──────────────────────────────
DATA_PATH = Path("data/processed/train_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WINDOW = 60       # 入力ウィンドウ（60分）
HORIZON = 1       # 15分後予測（1ステップ先）
BATCH  = 256
EPOCHS = 30
LR     = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── データ準備 ─────────────────────────
print("データ読み込み中...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# 正常データのみで学習
normal = df[df["anomaly"] == 0].copy()

# building_id=1 で動作確認
bid = 1
series = normal[normal["building_id"] == bid]["meter_scaled"].values
print(f"building_id={bid} 正常サンプル数: {len(series)}")

# スライディングウィンドウ（X: 過去60点 → y: 次の1点）
def make_windows(arr, window, horizon=1):
    X, y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_windows(series, WINDOW, HORIZON)
print(f"ウィンドウ数: {X.shape}, ラベル数: {y.shape}")

# LSTM入力形式: (batch, seq_len, features)
X_t = torch.tensor(X).unsqueeze(-1)
y_t = torch.tensor(y).unsqueeze(-1)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, shuffle=True)

# ── モデル定義 ─────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ── 学習 ───────────────────────────────
print("学習開始...")
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(loader):.6f}")

# ── 保存 ───────────────────────────────
torch.save(model.state_dict(), MODEL_DIR / "lstm_bid1.pth")
print(f"モデル保存: {MODEL_DIR}/lstm_bid1.pth")

# ── 予測誤差確認 ───────────────────────
model.eval()
with torch.no_grad():
    pred = model(X_t.to(DEVICE)).cpu().numpy().flatten()
    actual = y_t.numpy().flatten()
    errors = np.abs(actual - pred) / (np.abs(actual) + 1e-6)

print(f"LSTM_score 正常データ: mean={errors.mean():.6f}, std={errors.std():.6f}")