"""
# Author: Dean-98543
# Time: 11/3/2022  16:18
# File: utils.py
# Info: 
      1.
      2.
      3.
"""
import numpy as np
import pandas as pd
from collections import Counter

def cal_entropy(data, X=''):
    """: 计算熵值
    """
    entropy = lambda pi: -pi * np.log2(pi)
    if isinstance(data, pd.DataFrame):
        a = data[X].value_counts()/data.shape[0]
        return sum(entropy(a))

    elif isinstance(data, pd.Series):
        a = data.value_counts()/data.shape[0]
        return sum(entropy(a))

    elif isinstance(data, list):
        cls2cnt = Counter(data)
        return sum(entropy(v/len(data)) for k, v in cls2cnt.items())


def cal_condiEntropy(data, X, Y):
    """: 计算在已知随机变量X的分布的情况下，随机变量Y的信息熵是多少
    data: pd.DataFrame格式
    X：随机变量（已知分布）
    Y：随机变量
    """
    pi = data[X].value_counts() / data[X].shape[0]
    ei = data.groupby(X).apply(lambda dd:cal_entropy(dd, X=Y))
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


def split_dataframe(data, col):
    return dict(list(data.groupby(col)))


def choose_best_col(data, label):
    cols = [col for col in data.columns if col != label]
    max_value, best_col = float('-inf'), ''
    max_splited = None

    for col in cols:
        cur_infor_gain = cal_inforGain(data=data, A=col, D=label)
        if cur_infor_gain > max_value:
            max_value, best_col = cur_infor_gain, col
            max_splited = split_dataframe(data=data, col=col)

    return max_value, best_col, max_splited


class ID3Tree:
    class Node:
        def __init__(self, name):
            self.name = name
            self.connections = {}

        def connect(self, label, node):
            self.connections[label] = node

    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node("Root")

    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + "  " + "(" + str(connection) + ")")
            self.print_tree(child_node, tabs + "    ")

    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

    def construct(self, parent_node, parent_connection_label, input_data, columns):
        max_value, best_col, max_splited = choose_best_col(input_data[columns], self.label)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return

        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col]

        for splited_value, splited_data in max_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)


if __name__ == '__main__':
    df_lu = pd.read_csv("./example_data_luwei.csv", sep='\t')
    print(df_lu)
    print(choose_best_col(df_lu, 'play'))
    treel = ID3Tree(df_lu, 'play')
    treel.construct_tree()
    treel.print_tree(treel.root, "")

