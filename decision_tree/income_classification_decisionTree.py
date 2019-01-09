# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/5 20:16
"""
"""
决策树模型完成分类问题
"""
import pandas as pd
from sklearn import tree
from sklearn import model_selection

data = pd.read_csv('DecisionTree.csv')
feature_columns = [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race', u'gender', u'native-country']
label_column = ['income']
# 区分特征和目标列
features = data[feature_columns]
label = data[label_column]
# 特征处理
features = pd.get_dummies(features)

# 初始化一个决策树模型
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
# 用决策树分类器拟合数据
clf = clf.fit(features.values, label.values)
# 交叉验证
kfold = model_selection.KFold(n_splits=5, random_state=2019)
result = model_selection.cross_val_score(clf, features.values, label.values, cv=kfold)
print(result.mean())

