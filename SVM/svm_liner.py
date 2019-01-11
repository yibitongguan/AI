# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:33:45 2019

@author: Dcm
"""
import numpy as np
import matplotlib.pyplot as plt

""""
测试线性SVM模型
"""
from sklearn.datasets.samples_generator import make_blobs
# 训练集，测试集
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
#plt.figure(figsize=(8,8))
#plt.scatter(X[:, 0], X[:, 1], c=y, s=50);

from sklearn.svm import SVC
clf = SVC(kernel='linear')  # 线性核函数
clf.fit(X, y) # 拟合数据

# 可视化
def plot_svc_decision_function(clf, ax=None):
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 50)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 50)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            # 计算距离
            P[i, j] = clf.decision_function([[xi, yj]])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plot_svc_decision_function(clf)

"""
测试SVM核函数
"""
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
# 高斯核(rbf kernel)
clf = SVC(kernel='rbf')
clf.fit(X, y)

plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
plot_svc_decision_function(clf)
