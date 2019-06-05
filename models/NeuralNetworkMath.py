# coding: utf-8

import numpy as np


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
        ln_real_outputs = list(map(np.log, real_outputs))
        multiplied = np.multiply(ln_real_outputs, expected_outputs)
        return -sum(multiplied)

    @classmethod
    def regularization(cls, weights_matrices=[], _lambda=0.1, n_examples=1):
        regularization = 0
        for weights in weights_matrices:
            regularization += np.sum(np.square(weights))
        regularization = (_lambda / (2 * n_examples)) * regularization
        return regularization

