import numpy as np
import yfinance as yf
from scipy.special import logsumexp

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

# 変分ベイズ推論の実装
def variational_inference(observations, states, start_prob, trans_prob, emission_prob, num_iter=100):
    num_obs = len(observations)
    num_states = len(states)
    
    # 初期化
    alpha = np.zeros((num_obs, num_states))
    beta = np.zeros((num_obs, num_states))
    gamma = np.zeros((num_obs, num_states))
    xi = np.zeros((num_obs - 1, num_states, num_states))
    
    # 前向きアルゴリズム
    alpha[0, :] = start_prob * emission_prob[:, observations[0]]
    for t in range(1, num_obs):
        for j in range(num_states):
            alpha[t, j] = np.sum(alpha[t - 1, :] * trans_prob[:, j]) * emission_prob[j, observations[t]]
    
    # 後向きアルゴリズム
    beta[-1, :] = 1
    for t in range(num_obs - 2, -1, -1):
        for i in range(num_states):
            beta[t, i] = np.sum(trans_prob[i, :] * emission_prob[:, observations[t + 1]] * beta[t + 1, :])
    
    # 期待値ステップ
    for t in range(num_obs):
        gamma[t, :] = alpha[t, :] * beta[t, :]
        gamma[t, :] /= np.sum(gamma[t, :])
    
    for t in range(num_obs - 1):
        xi[t, :, :] = alpha[t, :, np.newaxis] * trans_prob * emission_prob[:, observations[t + 1]] * beta[t + 1, np.newaxis, :]
        xi[t, :, :] /= np.sum(xi[t, :, :])
    
    # 最大化ステップ
    for iteration in range(num_iter):
        start_prob = gamma[0, :]
        for i in range(num_states):
            for j in range(num_states):
                trans_prob[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        for j in range(num_states):
            for k in range(len(bins) - 1):
                emission_prob[j, k] = np.sum(gamma[observations == k, j]) / np.sum(gamma[:, j])
    
    return start_prob, trans_prob, emission_prob

start_prob, trans_prob, emission_prob = variational_inference(discrete_observations, states, start_prob, trans_prob, emission_prob)

print("Updated start probabilities:\n", start_prob)
print("Updated transition probabilities:\n", trans_prob)
print("Updated emission probabilities:\n", emission_prob)
