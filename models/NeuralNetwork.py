# coding: utf-8

import numpy as np
from models.InitialWeights import InitialWeights
from models.NeuralNetworkMath import NeuralNetworkMath


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
        return NeuralNetworkMath.sigmoid(self.outputs(features)).tolist()

    def outputs(self, features=[]):
        accumulator = np.array(features)
        # Multiplica todas as matrizes, (entrada x pesos) + bias.
        # Forward propagation
        for layer_i in range(len(self.weight_matrices)):
            accumulator = self.hidden_activation(accumulator, layer_i)
        return accumulator

    def hidden_activation(self, acc=[], layer=0):
        accumulator = np.array(acc)
        weights = self.weight_matrices[layer]
        bias = self.bias_weights_matrices[layer]
        return np.add(accumulator.dot(weights), bias)

    def train(self, training_dataset, _lambda=0.1):
        loss = 0
        for example in training_dataset.get_examples():
            # Isso tá incompleto. Ver a função NeuralNetworkMath.loss
            loss += NeuralNetworkMath.loss(self.outputs(example.get_body()))
        n_examples = len(training_dataset.examples)
        loss = loss / n_examples
        regularization = NeuralNetworkMath.regularization(self.weight_matrices,
                                                          _lambda=_lambda,
                                                          n_examples=n_examples)
        return loss + regularization

    def delta(self, deltas=[]):
        pass
