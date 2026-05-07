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
BATCH  = 256
EPOCHS = 30
LR     = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── データ準備 ─────────────────────────
print("データ読み込み中...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# 正常データのみで学習（One-class学習）
normal = df[df["anomaly"] == 0].copy()

# 建物1棟でまず動作確認（building_id=1）
bid = 1
series = normal[normal["building_id"] == bid]["meter_scaled"].values
print(f"building_id={bid} 正常サンプル数: {len(series)}")

# スライディングウィンドウ
def make_windows(arr, window):
    X = []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
    return np.array(X, dtype=np.float32)

X = make_windows(series, WINDOW)
print(f"ウィンドウ数: {X.shape}")

tensor = torch.tensor(X)
loader = DataLoader(TensorDataset(tensor), batch_size=BATCH, shuffle=True)

# ── モデル定義 ─────────────────────────
class AE(nn.Module):
    def __init__(self, input_dim=60):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ── 学習 ───────────────────────────────
print("学習開始...")
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        loss = criterion(out, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(loader):.6f}")

# ── 保存 ───────────────────────────────
torch.save(model.state_dict(), MODEL_DIR / "ae_bid1.pth")
print(f"モデル保存: {MODEL_DIR}/ae_bid1.pth")

# ── 異常スコア確認 ─────────────────────
model.eval()
with torch.no_grad():
    t = torch.tensor(X).to(DEVICE)
    recon = model(t)
    scores = ((t - recon)**2).mean(dim=1).cpu().numpy()

print(f"AE_score 正常データ: mean={scores.mean():.6f}, std={scores.std():.6f}")