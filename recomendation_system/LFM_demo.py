# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/9 15:51
"""
"""
编写了一个矩阵分解来完成一个LFM模型
"""
import numpy as np

# 矩阵分解
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    # 最长迭代次数
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        # 迭代公式计算
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        e = 0  # 损失值
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # 损失函数
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        # 加上正则化项
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T

# 测试，用P*Q还原矩阵补充未打分项
R = [
     [5,3,0,1],
     [4,0,3,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R = np.array(R)
print('R:', R)
N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N, K)
Q = np.random.rand(M, K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

print('nR:', nR)
