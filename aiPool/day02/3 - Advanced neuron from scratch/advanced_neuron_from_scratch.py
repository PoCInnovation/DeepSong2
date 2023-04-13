# Yours imports
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from pandas import DataFrame


def sigmoid(x):
    return lambda x: 1 / (1 + np.exp(-x))


def mean_squared_error(predict, target):
    if type(predict) == np.ndarray and type(target) == np.ndarray:
        assert len(predict) == len(target)
        return (1 / len(predict)) * np.sum(a=((predict - target) ** 2))
    else:
        return (predict - target) ** 2


class NeuralNetwork:
    def __init__(self, dimensions: list[int], lr: float, epochs: int) -> None:
        self.params = {}
        self.dimension_count = len(dimensions)
        self.lr = lr
        self.train_hist = np.zeros((epochs, 2))
        self.epochs = epochs
        self.params_count_halved = len(self.params) / 2

        for layer in range(self.dimension_count):
            self.params["W_" + str(layer)] = np.random.randn(dimensions[layer], dimensions[layer - 1])
            self.params["B_" + str(layer)] = np.random.randn(dimensions[layer], 1)


    def forward(self, X):
        return [sigmoid(x) for x in X]


    def backward(self, y, activations):
        np.dot()

    def update_weights(self):
        pass


def deep_neural_network_training(data_X, data_Y, layers=(16, 8, 16), learning_rate=0.1, epoch=500):
    network = NeuralNetwork(layers, learning_rate, epoch)


X, Y = make_circles(n_samples=1500, factor=0.3, noise=0.08, random_state=0)
deep_neural_network_training(X, Y)
