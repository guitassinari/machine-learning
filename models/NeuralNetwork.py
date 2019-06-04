# coding: utf-8

import numpy as np
from models.InitialWeights import InitialWeights


class NeuralNetwork:
    def __init__(self, layers_n_neurons=[], debug=False):
        self.n_layers = len(layers_n_neurons)
        self.weight_matrices = []
        self.bias_weights_matrices = []
        layer_neurons = layers_n_neurons
        for layer in range(self.n_layers):
            if layer == 0:
                continue
            previous_n_neurons = layer_neurons[layer - 1]
            n_neurons = layer_neurons[layer]
            weights = InitialWeights.generate(previous_n_neurons,
                                              n_neurons,
                                              debug=debug)
            bias_weights = InitialWeights.generate(1, n_neurons, debug=debug)
            self.weight_matrices.append(np.array(weights))
            self.bias_weights_matrices.append(bias_weights)

    def predict(self, features=[]):
        """
        :param features:
        :return:
        """
        accumulator = np.array(features)
        # Multiplica todas as matrizes, (entrada x pesos) + bias.
        # Forward propagation
        for layer_i in range(len(self.weight_matrices)):
            accumulator = self.hidden_activation(accumulator, layer_i)
        return self.sigmoid(accumulator).tolist()

    def hidden_activation(self, acc=[], layer=0):
        accumulator = np.array(acc)
        weights = self.weight_matrices[layer]
        bias = self.bias_weights_matrices[layer]
        return np.add(accumulator.dot(weights), bias)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def delta(self, deltas=[]):
        pass
