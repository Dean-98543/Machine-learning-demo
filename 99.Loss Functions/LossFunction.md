# 损失函数

## 高斯负对数似然损失(Gaussian Negative Log Likelihood Loss)

- torch官网：[torch.nn.GaussianNLLLoss()](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html)
- 真实标签服从高斯分布。神经网络（或者模型）的输出作为高斯分布的均值和方差
- 