import numpy as np
import matplotlib.pylab as pylab
# import matplotlib.pyplot as plt
import unittest
# from mpl_toolkits.mplot3d import axes3d
# 数值微分（求导）


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


class DrawDiff(unittest.TestCase):
    @unittest.skip
    def test_f1(self):
        x = np.arange(0.0, 20.0, 0.1)  # 以 0.1 为单位 0 到 20 的数组
        y = function_1(x)
        pylab.xlabel("x")
        pylab.ylabel("f(x)")
        pylab.plot(x, y)
        pylab.show()

    def test_f2(self):
        x0 = np.arange(-3.0, 3.0, 0.1)
        x1 = np.arange(-3.0, 3.0, 0.1)
        y = function_2(np.array([x0, x1]))
        pylab.xlabel("x")
        pylab.ylabel("f(x)")
        pylab.plot(x0, y)
        # pylab.show()


if __name__ == '__main__':
    unittest.main()
