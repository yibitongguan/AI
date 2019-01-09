# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/9 14:02
"""
"""
推荐算法demo
"""
# 构造一份虚拟打分数据集
users = {"小明": {"中国合伙人": 5.0, "太平轮": 3.0, "荒野猎人": 4.5, "老炮儿": 5.0, "我的少女时代": 3.0, "肖洛特烦恼": 4.5, "火星救援": 5.0},
         "小红": {"小时代4": 4.0, "荒野猎人": 3.0, "我的少女时代": 5.0, "肖洛特烦恼": 5.0, "火星救援": 3.0, "后会无期": 3.0},
         "小阳": {"小时代4": 2.0, "中国合伙人": 5.0, "我的少女时代": 3.0, "老炮儿": 5.0, "肖洛特烦恼": 4.5, "速度与激情7": 5.0},
         "小四": {"小时代4": 5.0, "中国合伙人": 3.0, "我的少女时代": 4.0, "匆匆那年": 4.0, "速度与激情7": 3.5, "火星救援": 3.5, "后会无期": 4.5},
         "六爷": {"小时代4": 2.0, "中国合伙人": 4.0, "荒野猎人": 4.5, "老炮儿": 5.0, "我的少女时代": 2.0},
         "小李":  {"荒野猎人": 5.0, "盗梦空间": 5.0, "我的少女时代": 3.0, "速度与激情7": 5.0, "蚁人": 4.5, "老炮儿": 4.0, "后会无期": 3.5},
         "隔壁老王": {"荒野猎人": 5.0, "中国合伙人": 4.0, "我的少女时代": 1.0, "Phoenix": 5.0, "甄嬛传": 4.0, "The Strokes": 5.0},
         "邻村小芳": {"小时代4": 4.0, "我的少女时代": 4.5, "匆匆那年": 4.5, "甄嬛传": 2.5, "The Strokes": 3.0}
        }

# 打分之间距离计算的函数
# 计算欧式距离
def euclidean_dis(rating1, rating2):
    dis = 0
    commonRatings = False
    for key in rating1:
        # 两个打分序列有公共打分电影才能计算距离
        if key in rating2:
            dis += (rating1[key] - rating2[key])**2
            commonRatings = True
    if commonRatings:
        return dis
    else:
        return -1

# 计算曼哈顿距离/L1距离
def manhattan_dis(rating1, rating2):
    dis = 0
    commonRatings = False
    for key in rating1:
        # 两个打分序列有公共打分电影才能计算距离
        if key in rating2:
            dis += abs(rating1[key] - rating2[key])
            commonRatings = True
    if commonRatings:
        return dis
    else:
        return -1

# 计算cos距离
from math import sqrt
def cos_dis(rating1, rating2):
    dis = 0
    commonRatings = False
    dot_product_1 = 0
    dot_product_2 = 0
    for score in rating1.values():
        dot_product_1 += score ^ 2
    for score in rating2.values():
        dot_product_2 += score ^ 2
    for key in rating1:
        if key in rating2:
            dis += rating1[key] * rating2[key]
            commonRatings = True
    if commonRatings:
        return 1 - dis / sqrt(dot_product_1 * dot_product_2)
    else:
        return -1

# 计算pearson距离
def pearson_dis(rating1, rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    # now compute denominator
    denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator

"""
给定username的情况下，计算其他user和该username的距离并排序
"""
def compute_near_neighbor(username, users):
    distances = []
    for user in users:
        if user != username:
            dis = pearson_dis(users[username], users[user])
            distances.append((dis, user))
    # 根据距离排序
    distances.sort()
    return distances

"""
完成topN推荐
"""
def recommend(username, users):
    nearest = compute_near_neighbor(username, users)[0][1]
    print('最近邻：', nearest)
    recommendations = []
    # 找到最近邻看过，但是我们没看过的电影，计算推荐
    neighborRatings = users[nearest]
    userRatings = users[username]
    for movie in neighborRatings:
        if not movie in userRatings:
            recommendations.append((movie, neighborRatings[movie]))
    results = sorted(recommendations, key=lambda movie: movie[1], reverse=True)
    for result in results:
        print(result[0], result[1])

# 测试
recommend('六爷', users)
