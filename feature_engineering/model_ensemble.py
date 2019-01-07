# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/7 22:18
"""
"""
模型融合
"""
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#年纪、怀孕、血液检查的次数... 匹马印第安人糖尿病的数据集
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('pima-indians-diabetes.data.csv', names=names)
# 准备数据
array = df.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = model_selection.KFold(n_splits=5, random_state=2019)

# 1.投票器模型融合
# 创建投票器子模型
from sklearn.ensemble import VotingClassifier
estimators = []
model_1 = LogisticRegression()
estimators.append(('logistic', model_1))
model_2 = DecisionTreeClassifier()
estimators.append(('dt', model_2))
model_3 = SVC()
estimators.append(('svm', model_3))
# 投票器融合
ensemble = VotingClassifier(estimators)
# 交叉验证
result = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(result.mean())

# 2.Bagging
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier()
num = 100
model = BaggingClassifier(base_estimator=dt, n_estimators=num, random_state=2019)
result = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# 3.RandomForst
from sklearn.ensemble import RandomForestClassifier
num_trees = 100
max_feature_num = 5
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_feature_num)
result = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
num_trees = 25
model = AdaBoostClassifier(n_estimators=num_trees, random_state=2019)
result = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(result.mean())
