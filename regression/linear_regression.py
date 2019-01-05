# encoding:utf-8
"""
@author = Dcm
@create_time = 2018/12/20 16:50
"""
"""
使用梯度下降完成线性回归
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
# 得到X、y的矩阵
# 特征x中增加一维x0=1，表示截距项
X = np.c_[np.ones(data.shape[0]), data[:, 0]]
y = np.c_[data[:, 1]]
# 画出散点图
# plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
# plt.xlim(4, 24)
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# #plt.show()

# 定义损失函数
def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    J = 0
    h = X.dot(theta)
    # 公式计算损失函数
    J = 1.0/(2*m)*(np.sum(np.square(h-y)))
    return J

# 梯度下降
def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        # 沿着负梯度的方向减小
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    # 得到结果值theta
    return (theta, J_history)

# 画出每一次迭代和损失函数变化
theta, Cost_J = gradientDescent(X, y)
# plt.plot(Cost_J)
# plt.ylabel('Cost J')
# plt.xlabel('Iterations')
# plt.show()

xx = np.arange(5, 23)
yy = theta[0]+theta[1]*xx

# 画出线性回归梯度下降收敛的情况
plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx, yy, label='Gradient descent')

# 和Scikit-learn中的线性回归对比一下
regr = LinearRegression()
regr.fit(X[:, 1].reshape(-1, 1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='sklearn.linear_model')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()
