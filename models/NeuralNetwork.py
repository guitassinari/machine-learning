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
        self.n_hidden_layers = self.n_layers - 2
        self.weight_matrices = []
        self.bias_weights_matrices = []
        self._lambda = hyper_parameters["lambda"] or 0.1
        self.batch_size = hyper_parameters["batch_size"] or 1
        self.last_activations = []
        self.deltas = []
        self.bias_deltas = []
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
        accumulator = np.array(features_matrix).astype(np.float)
        # Multiplica todas as matrizes, (entrada x pesos) + bias.
        # Forward propagation
        last_activations = []

        for layer_i in range(self.n_layers-1):
            accumulator = self.hidden_activation(accumulator, layer_i)
            last_activations.append(accumulator)

        self.last_activations = last_activations
        return accumulator

    def hidden_activation(self, acc_matrix=[], layer=0):
        weights = self.weight_matrices[layer]
        bias = self.bias_weights_matrices[layer]
        multiplied = weights.dot(acc_matrix)
        zs = np.add(multiplied, bias)
        activations = NeuralNetworkMath.sigmoid(zs)
        return activations

    def back_propagate(self, expected_outputs=[], activations=[]):
        last_activation = activations[-1]
        deltas = NeuralNetworkMath.output_delta(expected_outputs, last_activation)
        deltas_matrices = [deltas]
        last_delta = deltas.copy()
        bias_deltas_matrices = [deltas]
        for layer in reversed(range(self.n_hidden_layers)):
            weight_matrix = self.weight_matrices[layer+1]
            bias_matrix = self.bias_weights_matrices[layer+1]
            activation = activations[layer]
            deltas = NeuralNetworkMath.delta(activation,
                                             weight_matrix,
                                             last_delta)
            last_delta = deltas
            bias_deltas_matrices.insert(0, deltas)
            deltas_matrices.insert(0, deltas)
        self.deltas = deltas_matrices
        self.bias_deltas = bias_deltas_matrices

    def update_weights(self, old_weights=[], gradients_matrices=[], alpha=0.1):
        print("OLD", old_weights)
        print("NEW", gradients_matrices)
        alpha_gradients = list(map(lambda matrix: np.multiply(matrix, alpha), gradients_matrices))

        new_weights_matrices = NeuralNetworkMath.matrix_list_operation(
            np.subtract,
            old_weights,
            alpha_gradients
        )

        return new_weights_matrices

    def train(self, training_dataset):
        all_examples = training_dataset.get_examples()
        n_examples = len(all_examples)
        gradients = None
        bias_gradients = None
        batch_counter = 0
        for example_index in range(n_examples): # podemos adicionar alguma condição de parada aqui
            batch_counter += 1
            example = training_dataset.get_example_at(example_index)
            inputs = example.get_body()
            inputs = np.array(inputs).astype(np.float).tolist()
            expected_outputs = self.example_expected_output(example, self.training_set)
            self.outputs(inputs)
            self.back_propagate(expected_outputs=expected_outputs,
                                activations=self.last_activations)
            input_and_activations = [inputs] + self.last_activations
            new_gradients = NeuralNetworkMath\
                .all_gradients(activations_matrices=input_and_activations,
                               deltas_matrices=self.deltas)
            if gradients is None:
                gradients = new_gradients
                bias_gradients = self.bias_deltas
            else:
                gradients = NeuralNetworkMath.matrix_list_operation(
                    np.add,
                    gradients,
                    new_gradients
                )
                bias_gradients = np.add(bias_gradients, self.bias_deltas)

            # fim do batch ou último exemplo
            if batch_counter == self.batch_size or example_index == n_examples:
                batch_counter = 0
                regularization = NeuralNetworkMath.gradient_regularization(
                    weights_matrices=self.weight_matrices,
                    _lambda=self._lambda
                )

                gradients = NeuralNetworkMath.matrix_list_operation(
                    np.add,
                    gradients,
                    regularization
                )
                gradients = list(map(lambda matrix: np.divide(matrix, n_examples), gradients))
                self.weight_matrices = self.update_weights(old_weights=self.weight_matrices,
                                                           gradients_matrices=gradients,
                                                           alpha=self.alpha)

                bias_gradients = list(map(lambda matrix: np.divide(matrix, n_examples), bias_gradients))
                self.bias_weights_matrices = self.update_weights(old_weights=self.bias_weights_matrices,
                                                                 gradients_matrices=bias_gradients,
                                                                 alpha=self.alpha)

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


    def update_momentum(self, iterator, weights_moments=[]):
        return self.weights_matrices[iterator] - (self.alpha * self.weights_moments[iterator])

    def example_expected_output(self, example, dataset):
        possible_classes = dataset.get_uniq_classes()
        example_class = example.get_class()

        return list(map(lambda klass: [1] if klass == example_class else [0], possible_classes))
