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
        self.last_activations = [None] * self.n_layers
        self.deltas = [None] * self.n_layers
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
        print(weights, acc_matrix, bias)
        zs = np.add(weights.dot(acc_matrix), bias)
        print(zs)
        activations = NeuralNetworkMath.sigmoid(zs)
        self.last_activations[layer] = activations
        return activations

    def back_propagate(self, expected_outputs=[], mini_batch_size, run_momentum, momentum_term):
        batch_counter = 0
        for layer in reversed(range(self.n_layers)):  # ate criterio de parada?
            if layer == self.last_layer_index():
                deltas = self.outputs(expected_outputs)
            else:
                activations = self.last_activations[layer]
                weights = self.weight_matrices[layer]
                weights_bias = self.bias_weights_matrices[layer]
                next_layer_deltas = self.deltas[layer+1]
                deltas = NeuralNetworkMath.delta(activations,
                                                 weights,
                                                 next_layer_deltas)
                deltas_bias = NeuralNetworkMath.delta(activations,
                                                 weights_bias,
                                                 next_layer_deltas)
                # acumula em D(l=k) os gradientes com base no exemplo atual
                # fara isso para cada camada
                gradients = NeuralNetworkMath.all_gradients(activations,
                                                            next_layer_deltas)
                # aplica regularização alpha a apenas a pesos não bias
                gradients_reg = NeuralNetworkMath.gradient_regularization(self._lambda,
                                                                weights)
                # combina gradientes com regularização;
                # divide por #exemplos para calcular gradiente médio
                regularized_gradients = (1/len(self.training_set.examples)) * (gradients + gradients_reg)
                # atualiza pesos de cada camada com base nos gradientes

                # caso o mini_batch seja maior que 1 (diferente de estocástico)
                if mini_batch_size > 1:
                    if mini_batch_size != batch_counter:
                        # conta iterações do batch
                        batch_counter += 1
                        # acumula gradientes
                        regularized_gradients += regularized_gradients
                        gradients += gradients
                    else:
                        for weight in weights:
                            # atualiza pesos de acordo com a média dos gradientes
                            weights = weights - (alpha * (regularized_gradients/mini_batch_size))
                            # atualizando peso de bias
                            weights_bias = weights_bias - (alpha * (gradients/mini_batch_size))
                # usar momentum aqui
                else:
                    for weight in weights:
                        weights = weights - (alpha * regularized_gradients)
                        # atualizando peso de bias
                        weights_bias = weights_bias - (alpha * gradients)

            # atualiza cada camada da rede
            self.deltas[layer] = deltas
            self.weight_matrices[layer] = weights
            self.bias_weights_matrices[layer] = weights_bias
        return

    def output_deltas(self, output_matrix=[[]]):
        outputs = self.last_activations[self.last_layer_index()]
        return np.subtract(outputs, output_matrix)

    def train(self, training_dataset):
        loss = 0
        for example in training_dataset.get_examples():
            # Isso tá incompleto. Ver a função NeuralNetworkMath.loss
            inputs = example.get_body()
            outputs = self.outputs(inputs)
            loss += NeuralNetworkMath.loss(outputs, [[1]])
        n_examples = len(training_dataset.examples)
        loss = loss / n_examples
        regularization = NeuralNetworkMath.loss_regularization(self.weight_matrices,
                                                               _lambda=self._lambda,
                                                               n_examples=n_examples)
        return loss + regularization

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

    def numerical_verifier(epsilon, weights_matrices=[], gradients=[], expected_outputs=[]):
        # J(T1 -epsilon, T2,...) - J(t1+epsilon, T2,...)
        # _______________________________________________
        #                   2*epsilon

        # funcionalidade que permita, via linha de comando,
        # efetuar a verificação numérica do gradiente,
        # a fim de checar a corretude da implementação de cada grupo;
        numerical_grad = 0
        for index in weights_matrices:
            weights_minus = weights_matrices[:]  # coloa a matrix inteira para variavel
            weights_plus = weights_matrices[:]

            weights_minus[index] -= epsilon
            weights_plus[index] += epsilon
            numerical_grad = (loss(weights_plus, expected_outputs) - loss(weights_minus, expected_outputs)) / (2*epsilon)
            # Ideia aqui e printar duas colunas lado a lado
            # A primeira sera o valor calculado pela rede
            # A segunda o valor aproximado numericamente

            # Se der tempo vamos mostrar graficamente o diferenca
            print("Gradients: " gradients[row,col], numerical_grad[row,col])
        return

    def momentum(momentum_term, weight_moments=[], gradients=[], iterator):
        weight_moments[iterator] = (momentum_term * weight_moments) + gradients

        return
    def update_momentum(weights_matrices=[], weights_moments=[], alpha, iterator):
        return weights_matrices[iterator] - (alpha * weights_moments~[iterator])
