# encoding:utf-8
"""
@author = Dcm
@create_time = 2019/1/8 10:08
"""
"""
朴素贝叶斯预测
"""
import numpy as np
import pandas as pd

# 创建数据集
data = pd.DataFrame()

# 把Y定义好
data['Gender'] = ['male','male','male','male','female','female','female','female']
# 并给定他们的属性
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]
# 创建预测点
person = pd.DataFrame()
person['Height'] = [6]
person['Weight'] = [130]
person['Foot_Size'] = [8]

# 男人的总数
n_male = data['Gender'][data['Gender'] == 'male'].count()
# 女人的总数
n_female = data['Gender'][data['Gender'] == 'female'].count()
# 全部的人数
total_ppl = data['Gender'].count()

# 计算先验概率
P_male = n_male/total_ppl
P_female = n_female/total_ppl

# 利用高斯分布计算P(x|y)的概率
def p_x_given_y(x, mean_y, variance_y):
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p

data_means = data.groupby('Gender').mean()
data_variance = data.groupby('Gender').var()
# 男人的Means
male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
male_footsize_mean = data_means['Foot_Size'][data_variance.index == 'male'].values[0]
# 男人的Variance
male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]
# 女人的Means
female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
female_footsize_mean = data_means['Foot_Size'][data_variance.index == 'female'].values[0]
# 女人的Variance
female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]

# 测试数据是男人的概率
p_is_male = (P_male *
p_x_given_y(person['Height'][0], male_height_mean, male_height_variance) *
p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) *
p_x_given_y(person['Foot_Size'][0], male_footsize_mean, male_footsize_variance))
# 测试数据是女人的概率
p_is_female = (P_female *
p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) *
p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) *
p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance))

# 打印结果
print(p_is_male, p_is_female)
