import os
import sys
import unittest

import numpy as np
from ch3_02 import sigmoid, softmax
from ch4_01 import onehot_cross_entropy_error as cross_entropy_error
from gradient import numerical_gradient

sys.path.append(os.pardir)


class TwoLayerNet:
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01
    ) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x: np.ndarray):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):  # x输入数据，t监督数据
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t): # 太慢啦
        loss_W = lambda _: self.loss(x, t)  # noqa: E731

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


class TestLayer(unittest.TestCase):
    @unittest.skip
    def test_shape(self):
        net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
        print(net.params["W1"].shape)
        print(net.params["b1"].shape)
        print(net.params["W2"].shape)
        print(net.params["b2"].shape)

    def test_predict(self):
        net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
        x = np.random.rand(100, 784)
        y = net.predict(x)
        print(y)


if __name__ == "__main__":
    unittest.main()
