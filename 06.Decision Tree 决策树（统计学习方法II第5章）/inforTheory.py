"""
# Author: Dean-98543
# Time: 11/3/2022  16:18
# File: inforTheory.py
# Info: 
      1.
      2.
      3.
"""
import numpy as np
import pandas as pd

def cal_entropy(data, X):
    """: 用来计算随机变量X的信息熵
    data: pd.DataFrame格式
    X：要计算熵的数据所在的列名
    """
    a = data[X].value_counts()/data.shape[0]
    entropy = lambda pi: pi*np.log2(pi)
    return sum(-entropy(a))


def cal_condiEntropy(data, X, Y):
    """: 计算在已知随机变量X的分布的情况下，随机变量Y的信息熵是多少
    data: pd.DataFrame格式
    X：随机变量（已知分布）
    Y：随机变量
    """
    pi = data[X].value_counts() / data[X].shape[0]
    ei = data.groupby(X).apply(lambda xxx:cal_entropy(xxx, Y))
    condi_entropy = sum(pi*ei)
    return condi_entropy


def cal_inforGain(data, A, D):
    """:
    计算特征A对训练数据集D的信息增益g(D, A),
    定义为  集合D的经验熵H(D)  与  在给定特征A的条件下D的经验条件熵H(D|A)  之差

    data：pd.DataFrame格式
    A：特征
    D：训练数据集
    """
    return cal_entropy(data, D) - cal_condiEntropy(data, X=A, Y=D)


def cal_inforGainRatio(data, A, D):
    return cal_inforGain(data, A=A, D=D)/cal_entropy(data, A)


