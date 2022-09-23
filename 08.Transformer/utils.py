"""
# Author: Dean-98543
# Time: 8/1/2022  15:51
# File: utils.py
# Info: 
      1.
      2.
      3.
"""
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import random


def set_seed(seed=666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(20220801)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    # query.dim=4
    d_k = query.size(-1)        # d_model//head = 512//8 = 64
    # tmp = torch.matmul(query, key.transpose(-2, -1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn      # 权重矩阵，P矩阵


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)


if __name__ == "__main__":

    sm = subsequent_mask(5)
    print(sm)

    #q = torch.rand(size=(1, 8, 26, 64))     # 8: head numbers  64 = 512 // 8
    #k = torch.rand(size=(1, 8, 26, 64))
    #v = torch.rand(size=(1, 8, 26, 64))
    #y1, y2 = attention(query=q,
    #                   key=k,
    #                   value=v,
    #                   mask=None,
    #                   dropout=None)
    #print(y1.shape, y2.shape)
