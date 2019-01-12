# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
data = digits.data
label = digits.target
num_insts, num_feats = data.shape[0], data.shape[1]
# 打乱顺序
ridx = np.arange(num_insts)
np.random.shuffle(ridx)
data = data[ridx, :]
label = label[ridx]

num_clusters = 10
kmeans_centers = np.zeros((num_clusters, num_feats))
# 初始化：取10张随机的图片作为cluster中心
kmeans_centers = data[:num_clusters, :]

num_iters = 10
losses = np.zeros(num_iters)
start_time = time.time()
for i in range(num_iters):
    # Compute the distance of each instance to each cluster center. 
    inst_sqs = np.sum(data * data, axis=1, keepdims=True)
    center_sqs = np.sum(kmeans_centers * kmeans_centers, axis=1)
    inner_prod = np.dot(data, kmeans_centers.T)
    # Get an num_insts x num_centers matrix D. D_ij = ||x_i - u_j||^2
    # 计算距离
    dist = inst_sqs + center_sqs - 2 * inner_prod
    # Compute the loss in current iteration.
    losses[i] = np.mean(np.min(dist, axis=1))
    # Compute cluster assignment in current iteration.
    # 重分配：将每一个数据点分配至距离最近的中心
    cluster_ids = np.argmin(dist, axis=1)
    # Update cluster center.
    # 重拟合：根据新的分配重新计算聚类中心
    for j in range(num_clusters):
        d = data[cluster_ids == j, :]
        kmeans_centers[j, :] = np.mean(d, axis=0)
end_time = time.time()
print("Time used for Kmeans clustering =", end_time - start_time, "seconds.")

# 查看损失函数
# =============================================================================
# plt.figure()
# plt.plot(np.arange(num_iters), losses, "o-", lw=3)
# plt.grid(True)
# plt.title("Kmeans")
# plt.xlabel("Iteration Number")
# plt.ylabel("Objective Loss")
# plt.show()
# =============================================================================

# 查看每个聚类中心：0-9
for j in range(num_clusters):
    plt.figure()
    plt.gray()
    plt.matshow(kmeans_centers[j].reshape(8, 8))
    plt.show()
