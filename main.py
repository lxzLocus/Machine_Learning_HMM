import numpy as np

# サンプルデータの生成（実際の株価データを使用する場合はここにデータを読み込む）
np.random.seed(42)
observations = np.random.choice([0, 1], size=(100,))

# HMMのパラメータの定義
states = [0, 1]
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3],
                       [0.4, 0.6]])
emission_prob = np.array([[0.5, 0.5],
                          [0.1, 0.9]])

# 前向きアルゴリズムの実装
def forward(observations, states, start_prob, trans_prob, emission_prob):
    alpha = np.zeros((len(observations), len(states)))
    alpha[0, :] = start_prob * emission_prob[:, observations[0]]

    for t in range(1, len(observations)):
        for j in range(len(states)):
            alpha[t, j] = np.sum(alpha[t-1] * trans_prob[:, j]) * emission_prob[j, observations[t]]
    
    return alpha

alpha = forward(observations, states, start_prob, trans_prob, emission_prob)
print("Alpha:\n", alpha)
