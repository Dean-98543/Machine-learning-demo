"""
# Author: Dean-98543
# Time: 8/1/2022  14:52
# File: PositionalEncoding.py
# Info: 
      1.
      2.
      3.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)        # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model)
        )       # (d_model//2, )
        # tmp = position * div_term         # (max_len, d_model//2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # (1, max_len, d_model)

        self.register_buffer('pe', pe)


    def forward(self, x):
        # x.size(1): sequence length
        x += self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    PE = PositionalEncoding(d_model=512,
                            dropout=0.2,
                            max_len=5000)
    x = torch.rand(size=(2, 26, 512))
    y = PE(x)
    print(x.size(), x.size(1), y.shape)









