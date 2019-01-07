# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/6 12:52
"""
"""
特征工程处理
"""
import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')

# 数据预处理 填充缺失值
df['Age'].fillna(value=df['Age'].mean())

# 特征工程操作
# 1.幅度变化
# 对数变换
log_age = df['Age'].apply(lambda x: np.log(x))
df.loc[:, 'log_age'] = log_age
# 最大最小值缩放
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
fare_trans = mm_scaler.fit_transform(df[['Fare']])
df.loc[:, 'fare_trans'] = fare_trans
# 标准化
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
fare_std_trans = std_scaler.fit_transform(df[['Fare']])
df.loc[:, 'fare_std_trans'] = fare_std_trans
# 2.统计值
# 分位数
age_quarter_1 = df['Age'].quantile(0.25)
age_quarter_3 = df['Age'].quantile(0.75)
# 3.四则运算
df.loc[:, 'family_number'] = df['SibSp'] + df['Parch'] + 1
# 4.高次特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_fea = poly.fit_transform(df[['SibSp', 'Parch']])
# 5.离散化
# 等距切分
df.loc[:, 'fare_cut'] = pd.cut(df['Fare'], 5)
# 等频切分
df.loc[:, 'fare_qcut'] = pd.qcut(df['Fare'], 5)
# 6.OneHot encoding/独热向量编码
embarked_oht = pd.get_dummies(df[['Embarked']])
# 7.时间型
car_sales = pd.read_csv('car_data.csv')
car_sales.loc[:, 'date'] = pd.to_datetime(car_sales['date_t'])
# 取出星期几
car_sales.loc[:, 'week'] = car_sales['date'].dt.dayofweek
# 判断是否周末
car_sales.loc[:, 'is_weekend'] = car_sales['week'].apply(lambda x: 1 if (x==0 or x==6) else 0)
# 8.词袋模型
corpus = [
    'This is a very good class',
    'students are very very very good',
    'This is the third sentence',
    'Is this the last doc'
]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names(), X.toarray)
vec = CountVectorizer(ngram_range=(1, 3))
X_ngram = vec.fit_transform(corpus)
# print(vec.get_feature_names(), X_ngram.toarray)
# 9.TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_X = tfidf_vec.fit_transform(corpus)
# print(tfidf_vec.get_feature_names(), tfidf_X.toarray)
# 10.组合特征
# 借助条件去判断获取组合特征
df.loc[:, 'alone'] = (df['SibSp']==0)&(df['Parch']==0)

print(df.head())
