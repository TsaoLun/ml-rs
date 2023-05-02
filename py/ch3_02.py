import numpy as np
from dataset.mnist import load_mnist
import sys
import os
import pickle
import unittest
sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def get_data():
    (img_train, label_train), (img_test, label_test) = load_mnist(
        flatten=True, normalize=True, one_hot_label=False)
    return img_test, label_test


def init_network():
    with open(os.path.dirname(sys.argv[0]) + "/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


class TestNLP(unittest.TestCase):
    @unittest.skip
    def test_accuract(self):
        imgs, labels = get_data()
        network = init_network()
        accuracy_cnt = 0

        for i in range(len(imgs)):
            y = predict(network, imgs[i])
            p = np.argmax(y)
            if p == labels[i]:
                accuracy_cnt += 1
        print("\nAccuracy:" + str(float(accuracy_cnt) / len(imgs)))

    def test_batch(self):
        imgs, labels = get_data()
        network = init_network()

        batch_size = 100
        accuracy_cnt = 0

        for i in range(0, len(imgs), batch_size):
            img_batch = imgs[i:i+batch_size]
            y_batch = predict(network, img_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == labels[i:i+batch_size])
        print("\nAccuracy:" + str(float(accuracy_cnt) / len(imgs)))


if __name__ == '__main__':
    unittest.main()
