import numpy as np
import matplotlib.pylab as pylab

# import matplotlib.pyplot as plt
import unittest

# from mpl_toolkits.mplot3d import axes3d
# 数值微分（求导）


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


# 梯度计算
def numerical_gradient(f, x: np.ndarray):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成 x 形状的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


class DrawDiff(unittest.TestCase):
    @unittest.skip
    def test_f1(self):
        x = np.arange(0.0, 20.0, 0.1)  # 以 0.1 为单位 0 到 20 的数组
        y = function_1(x)
        pylab.xlabel("x")
        pylab.ylabel("f(x)")
        pylab.plot(x, y)
        pylab.show()

    @unittest.skip
    def test_f2(self):
        x0 = np.arange(-3.0, 3.0, 0.1)
        x1 = np.arange(-3.0, 3.0, 0.1)
        y = function_2(np.array([x0, x1]))
        pass

    def test_ng(self):
        print(numerical_gradient(function_2, np.array([3.0, 4.0])))
        print(numerical_gradient(function_2, np.array([0.0, 2.0])))

if __name__ == "__main__":
    unittest.main()
