# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/9 19:37
"""
from sklearn import tree
from sklearn import model_selection
from sklearn.datasets import load_iris
# 数据
iris = load_iris()
# 得到特征值和标签
iris_features = iris.data
iris_target = iris.target

# 初始化一个决策树模型
clf = tree.DecisionTreeClassifier(max_depth=4)
# 用决策树分类器拟合数据
clf = clf.fit(iris_features, iris_target)
print(clf)
# 交叉验证
kfold = model_selection.KFold(n_splits=5, random_state=2019)
result = model_selection.cross_val_score(clf, iris_features, iris_target, cv=kfold)
print(result.mean())

