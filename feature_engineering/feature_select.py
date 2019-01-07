# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/7 21:50
"""
"""
特征选择
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

# 1.过滤型
from sklearn.feature_selection import SelectKBest
X_new = SelectKBest(k=2).fit_transform(X, y)
print(X_new.shape)

# 2.包裹型
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
X_rfe = RFE(estimator=rf, n_features_to_select=2).fit_transform(X, y)
print(X_rfe.shape)

# 嵌入式
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_embed = model.transform(X)
print(X_embed.shape)
