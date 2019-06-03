# coding: utf-8

import math
import numpy as np


class NeuralNetwork:
    def __init__(self, n_inputs=1, n_outputs=1, n_hidden_layers=0, layers_n_neurons=[]):
        self.n_layers = n_hidden_layers + 2
        self.weight_matrices = []
        layer_neurons = [n_inputs] + layers_n_neurons + [n_outputs]
        for layer in range(self.n_layers):
            if layer == 0:
                continue
            previous_n_neurons = layer_neurons[layer - 1]
            n_neurons = layer_neurons[layer]
            weights = np.random.random((previous_n_neurons, n_neurons))
            self.weight_matrices.append(np.array(weights))
        print(self.weight_matrices)

    def predict(self, features=[]):
        """
        :param features: uma lista de valores numéricos (deve ser do mesmo
        tamanho de self.weights)
        :return:
        """
        accumulator = np.array(features)
        for layer in range(self.n_layers - 1):
            weights = self.weight_matrices[layer]
            accumulator = accumulator.dot(weights)
        return list(accumulator)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def delta(self, deltas=[]):
        pass
