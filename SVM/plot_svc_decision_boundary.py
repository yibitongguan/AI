# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:54:38 2019

@author: Dcm
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fig_id, tight_layout=True):
    """保存绘图"""
    path = os.path.join(fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()  # 紧致布局
    plt.savefig(path, format='png', dpi=300)

from sklearn.svm import SVC
from sklearn import datasets
# 数据集
iris = datasets.load_iris()
# 只取petal length, petal width两个特征
X = iris["data"][:, (2, 3)]
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)  # 拟合数据

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    """决策边界画图"""
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    # 决策边界 w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]  # 间隔
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    # 特征向量特殊标识
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    
if __name__=='__main__':
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
