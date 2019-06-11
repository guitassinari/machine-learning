# coding: utf-8

import numpy as np
from models.InitialWeights import InitialWeights
from models.NeuralNetworkMath import NeuralNetworkMath


class NeuralNetwork:
    def __init__(self, hyper_parameters, training_set, debug=False):
        layers_n_neurons = hyper_parameters["layers_structure"]
        self.training_set = training_set
        self.debug = debug
        self.layer_neurons = layers_n_neurons
        self.n_layers = len(layers_n_neurons)
        self.weight_matrices = []
        self.bias_weights_matrices = []
        self._lambda = hyper_parameters["lambda"] or 0.1
        self.batch_size = hyper_parameters["batch_size"] or 1
        self.last_activations = [None] * self.n_layers
        self.deltas = [None] * self.n_layers
        self.alpha = hyper_parameters["alpha"] or 0.1
        self.build_neurons()
        self.train(training_set)

    def predict(self, features=[]):
        """
        :param features:
        :return:
        """
        return self.outputs(features)

    def outputs(self, inputs=[]):
        features_matrix = self.array_to_matrix(inputs)
        accumulator = features_matrix
        # Multiplica todas as matrizes, (entrada x pesos) + bias.
        # Forward propagation
        for layer_i in range(len(self.weight_matrices)):
            accumulator = self.hidden_activation(accumulator, layer_i)
        return accumulator.tolist()

    def hidden_activation(self, acc_matrix=[], layer=0):
        weights = self.weight_matrices[layer]
        bias = self.bias_weights_matrices[layer]
        zs = np.add(weights.dot(acc_matrix), bias)
        activations = NeuralNetworkMath.sigmoid(zs)
        self.last_activations[layer] = activations
        return activations

    def back_propagate(self, expected_outputs=[]):
        for layer in reversed(range(self.n_layers)):  # ate criterio de parada?
            if layer == self.last_layer_index():
                deltas = self.output_deltas(expected_outputs)
            else:
                activations = self.last_activations[layer]
                weight_matrix = self.weight_matrices[layer]
                bias_matrix = self.bias_weights_matrices[layer]
                next_layer_deltas = self.deltas[layer+1]
                deltas = NeuralNetworkMath.delta(activations,
                                                 weight_matrix,
                                                 next_layer_deltas)
                deltas_bias = NeuralNetworkMath.delta(activations,
                                                      bias_matrix,
                                                      next_layer_deltas)

            # atualiza cada camada da rede
            self.deltas[layer] = deltas
            self.weight_matrices[layer] = weight_matrix
            self.bias_weights_matrices[layer] = bias_matrix


    def update_weights(self, gradients_matrices=[], alpha=0.1):
        new_weights_matrices = []
        for weight_index in range(len(self.weight_matrices)):
            gradient_matrix = gradients_matrices[weight_index]
            weight_matrix = self.weight_matrices[weight_index]
            alpha_gradient_matrix = np.multiply(alpha, gradient_matrix)
            new_weights = np.subtract(weight_matrix, alpha_gradient_matrix)
            new_weights_matrices.append(new_weights)
        self.weight_matrices = new_weights_matrices

    def output_deltas(self, output_matrix=[[]]):
        outputs = self.last_activations[self.last_layer_index()]
        return np.subtract(outputs, output_matrix)

    def train(self, training_dataset):
        all_examples = training_dataset.get_examples()
        n_examples = len(all_examples)
        gradients = self.weight_matrices.copy()
        gradients.fill(0)
        batch_counter = 0
        for example_index in range(n_examples):
            batch_counter += 1
            example = training_dataset.get_example_at(example_index)
            # Isso tá incompleto. Ver a função NeuralNetworkMath.loss
            inputs = example.get_body()
            outputs = self.outputs(inputs)
            expected_outputs = []
            self.back_propagate(expected_outputs=expected_outputs)
            new_gradients = NeuralNetworkMath\
                .all_gradients(activations_matrices=self.last_activations,
                               deltas_matrices=self.deltas)
            gradients = np.sum(gradients, new_gradients)

            # fim do batch ou último exemplo
            if batch_counter == self.batch_size or example_index == n_examples:
                batch_counter = 0
                regularization = NeuralNetworkMath.gradient_regularization(
                    weights_matrices=self.weight_matrices,
                    _lambda=self._lambda)
                gradients = np.sum(gradients, regularization)
                gradients = gradients/n_examples
                self.update_weights(list(gradients.tolist()), self.alpha)

    def build_neurons(self):
        for layer in range(self.n_layers):
            if layer == 0:
                continue
            previous_n_neurons = self.layer_neurons[layer - 1]
            n_neurons = self.layer_neurons[layer]
            weights = InitialWeights.generate(n_neurons,
                                              previous_n_neurons,
                                              debug=self.debug)
            bias_weights = InitialWeights.generate(n_neurons, 1, debug=self.debug)
            self.weight_matrices.append(np.array(weights))
            self.bias_weights_matrices.append(bias_weights)

    def last_layer_index(self):
        return self.n_layers-1

    def array_to_matrix(self, array=[]):
        return list(map(lambda inp: [inp], array))

    def numerical_verifier(self, epsilon, gradients=[], expected_outputs=[]):
        # J(T1 -epsilon, T2,...) - J(t1+epsilon, T2,...)
        # _______________________________________________
        #                   2*epsilon

        # funcionalidade que permita, via linha de comando,
        # efetuar a verificação numérica do gradiente,
        # a fim de checar a corretude da implementação de cada grupo;
        numerical_grad = 0
        weights_minus = self.weights_matrices[:]  # coloa a matrix inteira para variavel
        weights_plus = self.weights_matrices[:]
        for index in weights_plus:
            weights_minus[index] -= epsilon
            weights_plus[index] += epsilon
            numerical_grad = (self.loss(weights_plus, expected_outputs) - self.loss(weights_minus, expected_outputs)) / (2*epsilon)
            # Ideia aqui e printar duas colunas lado a lado
            # A primeira sera o valor calculado pela rede
            # A segunda o valor aproximado numericamente

            # Se der tempo vamos mostrar graficamente o diferenca
            # print("Gradients: " gradients[row,col], numerical_grad[row,col])
        return

    def momentum(self, momentum_term, iterator, gradients=[]):
        weight_moments = []
        weight_moments[iterator] = (momentum_term * self.weight_moments) + gradients

        return
    def update_momentum(self, iterator, weights_moments=[]):
        return self.weights_matrices[iterator] - (self.alpha * self.weights_moments[iterator])
