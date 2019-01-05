# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/4 19:01
"""
"""
逻辑回归：根据两门成绩预测出学生是否pass
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 读取数据打印信息
def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    # print('Dimensions: ', data.shape)
    # print(data[1:6, :])
    return(data)

# 画图
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 负样本
    neg = data[:, 2] == 0
    # 正样本
    pos = data[:, 2] == 1

    if axes is None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)  # 设置图例

data = loaddata('data1.txt', ',')
# plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
y = np.c_[data[:, 2]]
# 定义sigmoid函数
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# 定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        return np.inf
    return J[0]

# 梯度
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * X.T.dot(h - y)
    # 转换为一维
    return (grad.flatten())

initial_theta = np.zeros(X.shape[1])
# 最小化损失函数
res = minimize(costFunction, initial_theta, args=(X, y), jac=gradient, options={'maxiter':400})
# print(res)
# 预测（45,85）
print(sigmoid(np.array([1, 45, 85]).dot(res.x.T)))

# 画出决策边界
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# 求得预测通过的概率做为高度
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
# 绘制高度h=0.5的等高线作为边界
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plt.show()
