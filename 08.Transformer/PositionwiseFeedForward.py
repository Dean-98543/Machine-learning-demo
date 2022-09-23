"""
# Author: Dean-98543
# Time: 8/11/2022  17:47
# File: PositionwiseFeedForward.py
# Info: 
      1.
      2.
      3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PisitionwiseFeedForward(nn.Module):       # 对解码器端的输入做预处理
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PisitionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features=d_model,
                             out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff,
                             out_features=d_model)
        self.dropout = dropout


    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


