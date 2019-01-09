# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/9 22:34
"""
"""
使用surprise进行Twitter电影预测
"""
import os
import io
from surprise import KNNBaseline, Reader
from surprise import Dataset

file_path = os.path.expanduser('./ratings.dat')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep='::')
# 从文件读取数据
movie_data = Dataset.load_from_file(file_path, reader=reader)
# 计算电影和电影之间的相似度
print("process data and build dataset...")
trainset = movie_data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}

def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    file_name = (os.path.expanduser('./movies.dat'))
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('::')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

# 首先，用算法计算相互间的相似度
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)

# 获取电影名到电影id 和 电影id到电影名的映射
rid_to_name, name_to_rid = read_item_names()

# 拿出来Pirates of the Caribbean: Dead Men Tell No Tales这部电影对应的item id
movie_raw_id = name_to_rid["Pirates of the Caribbean: Dead Men Tell No Tales (2017)"]
movie_inner_id = algo.trainset.to_inner_iid(movie_raw_id)

# 找到最近的10个邻居
movie_neighbors = algo.get_neighbors(movie_inner_id, k=10)

# 从近邻的id映射回电影名称
movie_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors)
movie_neighbors = (rid_to_name[rid] for rid in movie_neighbors)
# 输出
print("The 10 nearest neighbors of <Pirates of the Caribbean: Dead Men Tell No Tales (2017)> are:")
for movie in movie_neighbors:
    print(movie)
# 对用户进行推荐
user_inner_id = 1000
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0], user_rating)
for movie in items:
    print(algo.predict(user_inner_id, movie, r_ui=5), rid_to_name[algo.trainset.to_raw_iid(movie)])
