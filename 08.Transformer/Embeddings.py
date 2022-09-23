"""
# Author: Dean-98543
# Time: 8/5/2022  11:56
# File: Embeddings.py
# Info: 
      1.
      2.
      3.
"""
import math
import torch
import torch.nn as nn
from utils import set_seed
set_seed(20220805)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


if __name__ == "__main__":
    Embed = Embeddings(d_model=512, vocab=10**4)
    x = torch.randint(low=0, high=10**4, size=(8, 26))
    print(x, x.shape)   # (8, 26)
    y = Embed(x)
    print(y, y.shape)   # (8, 26, 512)