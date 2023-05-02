import numpy as np
# 目标：识别图像 5
# 方案一，从图形提取特征量，再使用机器学习（根据不同问题人工考虑特征量 SIFT, HOG -> 机器学习 SVM, KNN）
# 方案二，通过神经网络直接通过原始数据进行端到端学习

# 评估机器学习：训练数据(监督数据)+测试数据 -> 评估泛化能力
# 评估深度学习：损失函数（当前状态 --> 权重参数调整 --> 评估）
# 损失函数：均方误差 MSE


def origin_mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)
# 损失函数：交叉熵误差，正确标签输出 y 的自然对数


def origin_cross_entropy_error(y, t):
    delta = 1e-7  # protect -inf
    return -np.sum(t * np.log(y + delta))

# one-hot with mini-batch


def onehot_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# label with mini-batch


def label_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
