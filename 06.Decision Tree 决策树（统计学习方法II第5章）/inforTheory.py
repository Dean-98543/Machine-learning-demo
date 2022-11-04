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


def cal_inforGain(data, A, D):
    """
    data：pd.DataFrame格式
    X：随机变量X（条件已知）
    Y：随机变量Y
    """
    pi = data[A].value_counts() / data.shape[0]
    ei = data.groupby(A).apply(lambda xx:cal_entropy(xx, D))
    condi_entropy = sum(pi*ei)
    return cal_entropy(data, D) - condi_entropy


def cal_inforGainRatio(data, A, D):
    return cal_inforGain(data, A=A, D=D)/cal_entropy(data, A)


