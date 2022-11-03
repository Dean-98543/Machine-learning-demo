"""
# Author: Dean-98543
# Time: 11/3/2022  14:52
# File: main.py
# Info: 
      1.
      2.
      3.
"""
from _KNN import KNN
if __name__ == '__main__':
    x = [[0, 0],
         [1, 1],
         [-1, -1],
         [10, 0],
         [11, 0],
         [12, 0], ]
    y = ['A', 'A', 'A', 'A', 'B', 'B']
    knn = KNN(x, y)
    r = knn.predict([7, 6], 2)
    print(r)

