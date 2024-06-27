import numpy as np
import yfinance as yf

# 期間を指定
start = '2024-01-01'
end = '2024-05-30'

# 株価データを指定期間で取得
financeData = yf.download('NVDA', start, end)

# 終値を格納
observations = financeData['Close'].values

# 観測値を離散化（例として3つのビンに分類）
bins = np.linspace(np.min(observations), np.max(observations), num=4)
discrete_observations = np.digitize(observations, bins) - 1

# HMMのパラメータの定義
states = [0, 1]
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_prob = np.array([[0.5, 0.5, 0.0, 0.0],  # 状態0からの観測確率
                          [0.1, 0.1, 0.4, 0.4]]) # 状態1からの観測確率

# 変分推論の実装例
def variational_inference(observations, states, start_prob, trans_prob, emission_prob, max_iter=100):
    # 仮の初期化
    alpha = np.zeros((len(observations), len(states)))
    beta = np.zeros((len(observations), len(states)))
    gamma = np.zeros((len(observations), len(states)))

    for _ in range(max_iter):
        # Eステップ
        alpha[0, :] = start_prob * emission_prob[:, observations[0]]
        for t in range(1, len(observations)):
            for j in range(len(states)):
                alpha[t, j] = np.sum(alpha[t-1] * trans_prob[:, j]) * emission_prob[j, observations[t]]

        beta[-1, :] = 1
        for t in range(len(observations)-2, -1, -1):
            for i in range(len(states)):
                beta[t, i] = np.sum(trans_prob[i, :] * emission_prob[:, observations[t+1]] * beta[t+1])

        # Mステップ
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        start_prob = gamma[0]
        for i in range(len(states)):
            for j in range(len(states)):
                trans_prob[i, j] = np.sum(gamma[:-1, i] * beta[1:, j] * emission_prob[j, observations[1:]])

        trans_prob /= np.sum(trans_prob, axis=1, keepdims=True)

    return gamma

gamma = variational_inference(discrete_observations, states, start_prob, trans_prob, emission_prob)
print("Gamma:\n", gamma)
