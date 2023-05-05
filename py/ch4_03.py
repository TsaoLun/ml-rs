import sys, os

sys.path.append(os.pardir)
import numpy as np
from ch3_02 import softmax
from ch4_01 import label_cross_entropy_error as cross_entropy_error
from ch4_02 import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 高斯分布初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
