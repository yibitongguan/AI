# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:30:01 2019

@author: Dcm
"""
import jieba
import pandas as pd
# 读取数据
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()
df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()
df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()
df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()
df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()
technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",
                      names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = filter(lambda x:len(x)>1, segs)
            segs = filter(lambda x:x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception:
            print(line)
            continue 

#生成训练数据
sentences = []
preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')
# 打乱数据集
import random
random.shuffle(sentences)
# 分成训练集、测试集
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
"""文本分类器class"""
class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4),
                                          max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

# 测试
text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)
print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))
print(text_classifier.predict('苹果 公司 有 新 的 发布 计划'))
print(text_classifier.score(x_test, y_test))
