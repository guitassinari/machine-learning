# coding: utf-8

import numpy as np
from itertools import zip_longest

class NeuralNetworkMath:
    def __init__(self):
        pass

    @classmethod
    def sigmoid(cls, x):
        return 1. / (1. + np.exp(-x))

    @classmethod
    def loss(cls, real_outputs=[], expected_outputs=[]):
        """
        :param real_outputs: predições para um exemplo
        :param expected_outputs: array de 0s e um único 1, onde o 1 deve estar
        na posição da classe correta da predição
        :return: Multi-class Cross-Entropy loss
        https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        """
        # logaritmo natural das predições
        # expected_outputs representa todos os neuroios de saida???
        ln_real_outputs = list(map(np.log, real_outputs))
        multiplied = np.multiply(ln_real_outputs, expected_outputs)
        return -sum(multiplied)

    @classmethod
    def loss_regularization(cls, weights_matrices=[], _lambda=0.1, n_examples=1):
        regularization = 0
        for weights in weights_matrices:
            regularization += np.sum(np.square(weights))
        regularization = (_lambda / (2 * n_examples)) * regularization
        return regularization

    @classmethod
    def gradient_regularization(cls, _lambda=0.1, weights_matrices=[]):
        regularization = []
        for matrix in weights_matrices:
            regularized_matrix = np.multiply(matrix, _lambda)
            regularization.append(regularized_matrix.tolist())
        return regularization

    @classmethod
    def all_gradients(cls, activations_matrices=[], deltas_matrices=[]):
        gradients = []
        for index in range(len(activations_matrices)-1):
            activation_matrix = activations_matrices[index]
            delta_matrix = deltas_matrices[index]
            gradients.append(cls.gradient(activation_matrix, delta_matrix))
        return gradients

    @classmethod
    def gradient(cls, activations_matrix=[], next_layer_deltas_matrix=[]):
        transposed_activation = np.array(activations_matrix).transpose()
        deltas_matrix = np.array(next_layer_deltas_matrix)
        return np.multiply(deltas_matrix, transposed_activation)

    @classmethod
    def output_delta(cls, output=[], expected_output=[]):
        return np.subtract(output, expected_output)

    @classmethod
    def delta(cls, activations=[], weights=[], next_layer_deltas=[]):
        np_activations = np.array(activations)
        np_weights = np.array(weights)
        np_deltas = np.array(next_layer_deltas)
        weighted_deltas = np_weights.transpose().dot(np_deltas)
        one_sub_activations = np.subtract(1, np_activations)
        wtf_activation = np.multiply(one_sub_activations, np_activations)
        return np.multiply(weighted_deltas, wtf_activation)

    @classmethod
    def regularized_gradients(cls,
                              activation_matrices=[],
                              deltas_matrices=[],
                              weights_matrices=[],
                              _lambda=0.1):
        gradients = cls.all_gradients(activation_matrices, deltas_matrices)
        reg_matrix = cls.gradient_regularization(_lambda, weights_matrices)
        return np.add(gradients, reg_matrix)

    @classmethod
    def matrix_list_operation(cls, operator, list_1=[], list_2=[]):
        results = []
        for matrix_a, matrix_b in zip_longest(list_1, list_2):
            result = operator(matrix_a, matrix_b)
            results.append(result)
        return results
