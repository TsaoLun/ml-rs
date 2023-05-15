import sys
import os
import unittest
import numpy as np
from ch3_02 import softmax
from ch4_01 import label_cross_entropy_error as cross_entropy_error
from gradient import numerical_gradient

sys.path.append(os.pardir)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 高斯分布初始化

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    def loss(self, x: np.ndarray, t: np.ndarray):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


class netTest(unittest.TestCase):
    @unittest.skip
    def test_loss(self):
        net = simpleNet()
        print(net.W)
        x = np.array([0.6, 0.9])
        p = net.predict(x)
        print(p)
        print(np.argmax(p))
        t = np.array([0, 0, 1])  # 正确解标签
        print(net.loss(x, t))

    def test_lambda(self):
        net = simpleNet()
        x = np.array([0.6, 0.9])
        t = np.array([0, 0, 1])  # 正确解标签
        f = lambda _: net.loss(x, t)
        dW = numerical_gradient(f, net.W)
        print(dW)


if __name__ == "__main__":
    unittest.main()
