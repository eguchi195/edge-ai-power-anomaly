import pandas as pd
import numpy as np
from pathlib import Path

# パス設定
DATA_DIR = Path("LEAD1.0データセット")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# データ読み込み
print("データ読み込み中...")
df = pd.read_csv(DATA_DIR / "train_features.csv", parse_dates=["timestamp"])

# 基本情報
print(f"行数: {len(df)}, 建物数: {df['building_id'].nunique()}")
print(f"異常率: {df['anomaly'].mean():.4f}")
print(f"meter_readingのNaN率: {df['meter_reading'].isna().mean():.4f}")

# NaN処理：建物ごとに線形補間→前後埋め
print("NaN補間中...")
df = df.sort_values(["building_id", "timestamp"])
df["meter_reading"] = (
    df.groupby("building_id")["meter_reading"]
    .transform(lambda x: x.interpolate(method="linear").ffill().bfill())
)

# 負値をゼロに
df["meter_reading"] = df["meter_reading"].clip(lower=0)

# 建物ごとにMin-Maxスケーリング
print("スケーリング中...")
def minmax_scale(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return x * 0
    return (x - mn) / (mx - mn)

df["meter_scaled"] = (
    df.groupby("building_id")["meter_reading"]
    .transform(minmax_scale)
)

# 必要カラムのみ保存
out = df[["building_id", "timestamp", "meter_reading", "meter_scaled", "anomaly"]]
out.to_csv(OUT_DIR / "train_clean.csv", index=False)
print(f"保存完了: {OUT_DIR}/train_clean.csv")
print(out.describe())