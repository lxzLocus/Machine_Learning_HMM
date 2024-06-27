import numpy as np
import yfinance as yf

# 期間を指定
start = '2024-01-01'
end = '2024-05-30'

# 株価データを指定期間で取得
financeData = yf.download('NVDA', start, end)

# 終値を格納
observations = financeData['Close'].values

# データの確認
if observations.size == 0:
    raise ValueError("観測データが空です。")

# 観測値を離散化（例として3つのビンに分類）
bins = np.linspace(np.min(observations), np.max(observations), num=4)
discrete_observations = np.digitize(observations, bins) - 1

# HMMのパラメータの定義
states = [0, 1]
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_prob = np.array([[0.5, 0.5, 0.0, 0.0],  # 状態0からの観測確率
                          [0.1, 0.1, 0.4, 0.4]]) # 状態1からの観測確率

# 前向きアルゴリズムの実装
def forward(observations, states, start_prob, trans_prob, emission_prob):
    alpha = np.zeros((len(observations), len(states)))
    alpha[0, :] = start_prob * emission_prob[:, observations[0]]

    for t in range(1, len(observations)):
        for j in range(len(states)):
            alpha[t, j] = np.sum(alpha[t-1] * trans_prob[:, j]) * emission_prob[j, observations[t]]
    
    return alpha

alpha = forward(discrete_observations, states, start_prob, trans_prob, emission_prob)
print("Alpha:\n", alpha)
