import numpy as np
import pandas as pd
from pathlib import Path

# =============================================
# Fault Injection - 設備故障パターン生成
# 文献根拠：
#   冷蔵庫: Springer Neural Computing 2025
#   扇風機: Motor Current Signature Analysis (MCSA)
#   PC:     PSU Failure Analysis 2025
#   プリンター: USPTO特許 10761464
#   電子レンジ: Appliantology / Quora Magnetron Analysis
# =============================================

np.random.seed(42)
OUT_DIR = Path("data/fault")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 24
N_SAMPLES   = 1000  # 各パターンのサンプル数

# -----------------------------------------------
# 設備故障パターン（機器ごと）
# -----------------------------------------------

def fault_refrigerator(n, window):
    """冷蔵庫：コンプレッサー故障
    ON/OFFサイクルが乱れ、電力が急上昇→急低下を繰り返す
    """
    samples = []
    for _ in range(n):
        x = np.zeros(window)
        for i in range(window):
            if np.random.rand() < 0.6:  # 正常より高い頻度でON
                x[i] = np.random.uniform(0.7, 1.0)  # 正常は0.5前後
            else:
                x[i] = np.random.uniform(0.0, 0.2)  # 不規則なOFF
        samples.append(x)
    return np.array(samples)

def fault_fan(n, window):
    """扇風機：ベアリング摩耗
    消費電力が時間とともに緩やかに増加
    """
    samples = []
    for _ in range(n):
        base = np.random.uniform(0.3, 0.4)
        trend = np.linspace(0, np.random.uniform(0.2, 0.4), window)
        noise = np.random.normal(0, 0.02, window)
        x = np.clip(base + trend + noise, 0, 1)
        samples.append(x)
    return np.array(samples)

def fault_pc(n, window):
    """デスクトップPC：電源ユニット劣化
    消費電力に高周波ノイズが重畳し不規則に変動
    """
    samples = []
    for _ in range(n):
        base = np.random.uniform(0.4, 0.6)
        noise = np.random.normal(0, 0.1, window)  # 大きなノイズ
        x = np.clip(base + noise, 0, 1)
        samples.append(x)
    return np.array(samples)

def fault_printer(n, window):
    """プリンター：定着ユニット故障
    ウォームアップ時に電力が断続的に変動
    """
    samples = []
    for _ in range(n):
        x = np.zeros(window)
        for i in range(window):
            if i < 5:  # ウォームアップ期間
                x[i] = np.random.choice(
                    [np.random.uniform(0.8, 1.0),
                     np.random.uniform(0.1, 0.3)],
                    p=[0.5, 0.5]
                )
            else:
                x[i] = np.random.uniform(0.2, 0.5)
        samples.append(x)
    return np.array(samples)

def fault_microwave(n, window):
    """電子レンジ：マグネトロン劣化
    消費電力が増加しつつ断続的に変動
    """
    samples = []
    for _ in range(n):
        base = np.random.uniform(0.7, 0.9)  # 正常より高い消費
        x = np.array([
            base + np.random.choice([-0.2, 0.2]) * np.random.rand()
            for _ in range(window)
        ])
        x = np.clip(x, 0, 1)
        samples.append(x)
    return np.array(samples)

# -----------------------------------------------
# センサー誤計測パターン
# -----------------------------------------------

def sensor_spike(n, window):
    """スパイク：瞬間的に異常値が出現"""
    base = np.random.uniform(0.2, 0.6, (n, window))
    for i in range(n):
        idx = np.random.randint(0, window)
        base[i, idx] = np.random.uniform(0.9, 1.0)
    return base

def sensor_dropout(n, window):
    """欠損：データが一定期間ゼロになる"""
    base = np.random.uniform(0.2, 0.6, (n, window))
    for i in range(n):
        start = np.random.randint(0, window - 3)
        length = np.random.randint(2, 5)
        base[i, start:start+length] = 0.0
    return base

def sensor_stuck(n, window):
    """固着：同じ値がずっと続く"""
    samples = []
    for _ in range(n):
        val = np.random.uniform(0.2, 0.8)
        x = np.full(window, val)
        x += np.random.normal(0, 0.001, window)  # 微小ノイズ
        samples.append(x)
    return np.array(samples)

def sensor_noise(n, window):
    """ノイズ：全体にランダムなゆらぎが重畳"""
    base = np.random.uniform(0.2, 0.6, (n, window))
    noise = np.random.normal(0, 0.15, (n, window))
    return np.clip(base + noise, 0, 1)

# -----------------------------------------------
# 生成・保存
# -----------------------------------------------

fault_generators = {
    "fault_refrigerator": fault_refrigerator,
    "fault_fan":          fault_fan,
    "fault_pc":           fault_pc,
    "fault_printer":      fault_printer,
    "fault_microwave":    fault_microwave,
    "sensor_spike":       sensor_spike,
    "sensor_dropout":     sensor_dropout,
    "sensor_stuck":       sensor_stuck,
    "sensor_noise":       sensor_noise,
}

all_X, all_y = [], []

for label, (name, func) in enumerate(fault_generators.items()):
    X = func(N_SAMPLES, WINDOW_SIZE)
    all_X.append(X)
    all_y.extend([name] * N_SAMPLES)
    np.save(OUT_DIR / f"{name}.npy", X)
    print(f"生成完了: {name} → {X.shape}")

X_all = np.vstack(all_X)
y_all = np.array(all_y)
np.save(OUT_DIR / "fault_X.npy", X_all)
np.save(OUT_DIR / "fault_y.npy", y_all)
print(f"\n全パターン保存完了: {X_all.shape}")
print(f"ラベル種類: {np.unique(y_all)}")