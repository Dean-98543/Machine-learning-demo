"""
# Author: Dean-98543
# Time: 11/3/2022  16:18
# File: informationGain.py
# Info: 
      1.
      2.
      3.
"""
import numpy as np
import pandas as pd

def cal_entropy(data, feature):
    a = data[feature].value_counts()/data.shape[0]
    return sum(-a*np.log2(a))


def cal_inforGain(data, D, A):
    p1 = data[A].value_counts() / data.shape[0]
    e1 = data.groupby(A).apply(lambda x:cal_entropy(x, D))
    cond_enrtopy = sum(p1*e1)
    return cal_entropy(data, D) - cond_enrtopy


def cal_inforGainRatio(data, D, A):
    return cal_inforGain(data, D=D, A=A)/cal_entropy(data, A)


if __name__ == "__main__":
    df = pd.DataFrame(
        data={'年龄': ['专科', '专科', '专科', '专科', '专科', '本科', '本科', '本科', '本科', '本科', '研究生', '研究生', '研究生', '研究生', '研究生'],
              '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
              '有房': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
              '信贷情况': ['中', '高', '高', '中', '中', '中', '高', '高', '很高', '很高', '很高', '高', '高', '很高', '中'],
              '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']},
        columns=['年龄', '有工作', '有车', '信贷情况', '类别']
    )

    print(cal_inforGain(data=df, A='学历', D='类别'))
    print(cal_inforGainRatio(data=df, A='学历', D='类别'))

    print(cal_inforGain(data=df, A='婚否', D='类别'))
    print(cal_inforGainRatio(data=df, A='婚否', D='类别'))

    print(cal_inforGain(data=df, A='是否有车', D='类别'))
    print(cal_inforGainRatio(data=df, A='是否有车', D='类别'))

    print(cal_inforGain(data=df, A='收入水平', D='类别'))
    print(cal_inforGainRatio(data=df, A='收入水平', D='类别'))
