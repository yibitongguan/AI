# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:54:38 2019

@author: Dcm
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.svm import SVC
from sklearn import datasets
# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=ChangedBehaviorWarning)
# 数据集
iris = datasets.load_iris()
iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'

X = iris["data"][:, (0, 1)]
y = iris["target"]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

clf = SVC(C=1, kernel='rbf', gamma=0.1)
clf.fit(x_train, y_train)

print (clf.score(x_train, y_train)) 
print ('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
print (clf.score(x_test, y_test))
print ('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

# 画图
N = 500
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 第0列的范围
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()  # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
grid_show = np.dstack((x1.flat, x2.flat))[0] # 测试点
grid_hat = clf.predict(grid_show)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

cm_light = mpl.colors.ListedColormap(['#00FFCC', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='k', s=100, cmap=cm_dark)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)
