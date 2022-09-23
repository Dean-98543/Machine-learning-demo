"""
# Author: Dean-98543
# Time: 8/1/2022  16:00
# File: MultiHeadedAttention.py
# Info: 
      1.
      2.
      3.
"""
import torch
import torch.nn as nn
from utils import clones, attention, set_seed
set_seed(20220801)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h     # self.d_k = d_model//h = 512//8 = 64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.linears是四个堆叠起来的输入输出都是512的全连接层
        # 四个全连接层，分别对应query，key，value的权重层和最后的output层？

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):        # 形成Q，K，V矩阵的过程
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # query:(nbatches, sequence_length, 512), as are key and value
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]       # query:(nbatches, 8, sequence_length, 64), as are key and value

        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask = mask,
                                 dropout=self.dropout)       # x: (nbatches, 8, sequence_length, 64)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h*self.d_k)
        )       # x: (nbatches, sequence_length, 512)

        del query
        del key
        del value
        return self.linears[-1](x)


if __name__ == "__main__":
    MHA = MultiHeadedAttention(h=8,
                               d_model=512,
                               dropout=0.2)
    x = torch.rand(size=(1, 26, 512))
    y = MHA(x, x, x, mask=None)
    print(y)
    print(y.shape)


