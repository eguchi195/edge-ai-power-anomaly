import numpy as np
import torch
from pathlib import Path

# =============================================
# 3クラス統合スコア
# AEスコア × LSTMスコアの組み合わせで分類
#
# 分類ロジック：
#   正常        : AE低 & LSTM低
#   設備故障     : AE高 & LSTM高
#   センサー誤計測: AE高 & LSTM低
# =============================================

# 閾値（事前実験で決定予定・暫定値）
THETA_AE   = 0.01
THETA_LSTM = 0.10

def classify(ae_score: float, lstm_score: float) -> str:
    ae_high   = ae_score   > THETA_AE
    lstm_high = lstm_score > THETA_LSTM

    if not ae_high and not lstm_high:
        return "normal"
    elif ae_high and lstm_high:
        return "device_fault"
    elif ae_high and not lstm_high:
        return "sensor_fault"
    else:
        # AE低・LSTM高（第4象限）→判定困難
        return "unknown"

def batch_classify(ae_scores: np.ndarray,
                   lstm_scores: np.ndarray) -> np.ndarray:
    return np.array([
        classify(a, l)
        for a, l in zip(ae_scores, lstm_scores)
    ])

if __name__ == "__main__":
    # 動作確認：各クラスのサンプルで分類テスト
    test_cases = [
        (0.001, 0.05, "normal"),
        (0.050, 0.30, "device_fault"),
        (0.050, 0.05, "sensor_fault"),
        (0.001, 0.30, "unknown"),
    ]

    print("=== 統合スコア分類テスト ===")
    all_pass = True
    for ae, lstm, expected in test_cases:
        result = classify(ae, lstm)
        ok = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"{ok} AE={ae:.3f} LSTM={lstm:.3f} → {result} (期待値: {expected})")

    print(f"\n{'全テスト通過' if all_pass else '一部テスト失敗'}")
    print(f"閾値: THETA_AE={THETA_AE}, THETA_LSTM={THETA_LSTM}")