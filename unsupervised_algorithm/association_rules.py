# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:31:55 2019

@author: Dcm
"""
"""基于关联规则的商品关联挖掘"""
import time
# 购买记录
transactions = [["milk", "bread"], 
                ["butter"], 
                ["beer", "diapers"],
                ["milk", "bread", "butter"],
                ["bread"]
               ]
items = set([t for transaction in transactions for t in transaction])
num_items = len(items)
item2id = {t:i for t, i in zip(items, range(num_items))}
id2item = [t for t in items]

# 定义数字表示
def init2items(s, num_items):
    items = []
    for i in range(num_items):
        if s % 2:
            items.append(id2item[num_items-1-i])
            s >>= 1
        return items
 
def items2int(items, num_items):
    idx = 0
    for t in items:
        idx += 2 ** item2id[t]
    return idx

def is_subset(sx, sy):
    """判断是否为子集"""
    for x in sx:
        if x not in sy:
            return False
    return True
    
def support(s, transactions):
    """计算物品出现频率"""
    num_trans = len(transactions)
    counter = 0
    for transaction in transactions:
        if is_subset(s, transaction): counter += 1
    return counter / num_trans

# =============================================================================
# # 寻找频繁集,暴力穷举
# sup = 0.3  # 阈值
# num_transactions = len(transactions)
# freq_sets = []
# start_time = time.time()
# for i in range(1, 2**num_transactions):
#     subsets = int2items(i, num_items)
#     counter = 0
#     freq = support(subsets, transactions)
#     if freq > sup: freq_sets.append(subsets)
# end_time = time.time()
# print("Time used for mining:", end_time - start_time, "seconds.")
# =============================================================================

# 寻找频繁集，Apriori算法，BFS广度优先
sup = 0.3
freq_sets = []
searched = set()
queue = []
head, tail = 0, 0
queue.append([])
searched.add(0)
tail += 1
start_time = time.time()
while head < tail:
    top = queue[head]
    head += 1
    for t in items:
        if t in top: continue
        curset = top + [t]
        idx = items2int(curset, num_items)
        freq = support(curset, transactions)
        if freq > sup and idx not in searched:
            # Is a frequent itemset, add to the queue, otherwise stop searching.
            freq_sets.append(curset)
            queue.append(curset)
            searched.add(idx)
            tail += 1
end_time = time.time()
print("All frequent subsets:", freq_sets)
print("Time used for mining:", end_time - start_time, "seconds.")