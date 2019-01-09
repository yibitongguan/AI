# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/9 20:45
"""
"""
使用随机森林预测房价
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn import model_selection
# 数据
boston_house = load_boston()
# 得到特征值和标签
features = boston_house.data
target = boston_house.target

# 初始化一个回归树模型
clf = RandomForestRegressor(n_estimators=15)
# 用决策树分类器拟合数据
clf = clf.fit(features, target)
print(clf)
# 交叉验证
kfold = model_selection.KFold(n_splits=5, random_state=2019)
result = model_selection.cross_val_score(clf, features, target, cv=kfold)
print(result.mean())
