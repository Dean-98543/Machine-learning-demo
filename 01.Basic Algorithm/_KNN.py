"""
# Author: Dean-98543
# Time: 11/3/2022  14:52
# File: _KNN.py
# Info: 
      1.
      2.
      3.
"""
import numpy as np

class KNN(object):
    """:cvar
    exp_x: 样本的特征向量集
    exp_y: 对应的标签集
    weights: 每个特征对应的权重
    distance_func: 用于计算距离的函数,
        distance_func(simple_feature_vector,target_feature_vector)
    """
    def __init__(self, exp_x, exp_y, weights=1, distance_func=None):
        self.__weights = weights
        if not isinstance(exp_x, np.ndarray):
            self.__x = np.array(exp_x)
        self.__y = exp_y

        assert len(self.__x)==len(self.__y), 'Number of x and y must be equal!'
        self.num_samples = len(self.__x)
        self.num_classes = len(set(self.__y))

        if distance_func is None:
            self.__distance_func = self.default_dist
        else:
            self.__distance_func = distance_func


    def default_dist(self, vector, target_vector, weights):
        return np.sum(((target_vector - vector) * weights) ** 2) ** 0.5


    def predict(self, target_vector, k=5):
        """:cvar
        target_vector: 需要预测的特征向量
        k: 结果取前k个
        return: 返回预测的标签
        """
        assert k < self.num_samples, 'The k must be less the length of all samples'

        if not isinstance(target_vector, np.ndarray):
            target_vector = np.array(target_vector)

        # 将所有距离的结果放入一个列表中，如果数据量比较大则需要使用数据库进行存储
        all_dist = [(None, None)]*self.num_samples
        for i, (vector, tag) in enumerate(zip(self.__x, self.__y)):
            ds = self.__distance_func(vector, target_vector, self.__weights)
            all_dist[i] = (ds, tag)

        # 根据distance对所有样本点进行排序
        all_dist.sort(key=lambda x: x[0])
        result_list = all_dist[:k + 1]
        freq = {}  # 统计标签频率
        for ds, tag in result_list:
            freq[tag] = freq.get(tag, 0) + 1
        return max(freq.items(), key=lambda x: x[1])[0]  # 返回频率高的标签



