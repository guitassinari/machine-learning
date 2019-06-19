# coding: utf-8

import numpy as np
from models.InitialWeights import InitialWeights
from models.NeuralNetworkMath import NeuralNetworkMath


class NeuralNetwork:
    def __init__(self, hyper_parameters, training_set, initial_weights=None, debug=False):
        n_inputs = len(training_set.get_attr_names())
        if training_set.class_data_type() == float:
            n_outputs = 1
        else:
            n_outputs = len(training_set.get_uniq_classes())
        layers_n_neurons = [n_inputs] + hyper_parameters["layers_structure"] + [n_outputs]
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
        self.alpha = hyper_parameters["alpha"] or 0.1
        self.build_neurons()
        if initial_weights is not None:
            self.weight_matrices = initial_weights
        self.train(training_set)

    def predict(self, example):
        """
        :param example:
        :return:
        """
        features = example.get_body()
        outputs = self.outputs(features)
        max_index = np.array(outputs).argmax()
        return self.training_set.get_uniq_classes()[max_index]

    def outputs(self, inputs=[]):
        features_matrix = NeuralNetworkMath.array_to_matrix(inputs)
        # print("FEAT MATRIX", features_matrix)
        accumulator = np.array(features_matrix).astype(np.float)
        # Multiplica todas as matrizes, (entrada x pesos) + bias.
        # Forward propagation

        self.last_activations = NeuralNetworkMath.all_activations_for(
            inputs=accumulator,
            weights_matrices=self.weight_matrices,
            bias_matrices=self.bias_weights_matrices)
        return self.last_activations[-1]

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
            expected_outputs = NeuralNetworkMath.example_expected_output(example, self.training_set)
            outputs = self.outputs(inputs)
            deltas, bias_deltas = NeuralNetworkMath.calculate_deltas(weights_matrices=self.weight_matrices,
                                                                     expected_outputs=expected_outputs,
                                                                     activations=self.last_activations)
            new_gradients = NeuralNetworkMath.all_gradients(
                activations_matrices=self.last_activations,
                deltas_matrices=deltas)
            if gradients is None:
                gradients = new_gradients
                bias_gradients = bias_deltas
            else:
                gradients = NeuralNetworkMath.matrix_list_operation(
                    np.add,
                    gradients,
                    new_gradients
                )
                bias_gradients = NeuralNetworkMath.matrix_list_operation(
                    np.add,
                    bias_gradients,
                    bias_deltas
                )

            # fim do batch ou último exemplo
            if batch_counter == self.batch_size or example_index == n_examples:
                batch_counter = 0
                regularization = NeuralNetworkMath.gradients_regularization(
                    weights_matrices=self.weight_matrices,
                    _lambda=self._lambda
                )

                gradients = NeuralNetworkMath.matrix_list_operation(
                    np.add,
                    gradients,
                    regularization
                )
                gradients = list(map(lambda matrix: np.divide(matrix, n_examples), gradients))
                self.weight_matrices = NeuralNetworkMath.update_weights(
                    old_weights=self.weight_matrices,
                    gradients_matrices=gradients,
                    alpha=self.alpha)

                bias_gradients = list(map(lambda matrix: np.divide(matrix, n_examples), bias_gradients))
                self.bias_weights_matrices = NeuralNetworkMath.update_weights(
                    old_weights=self.bias_weights_matrices,
                    gradients_matrices=bias_gradients,
                    alpha=self.alpha)

    def build_neurons(self):
        for layer in range(self.n_layers):
            if layer == 0:
                continue
            previous_n_neurons = int(self.layer_neurons[layer - 1])
            n_neurons = int(self.layer_neurons[layer])
            weights = InitialWeights.generate(n_neurons,
                                              previous_n_neurons,
                                              debug=self.debug)
            bias_weights = InitialWeights.generate(n_neurons, 1, debug=self.debug)
            self.weight_matrices.append(np.array(weights))
            self.bias_weights_matrices.append(bias_weights)

    def last_layer_index(self):
        return self.n_layers-1

    def momentum(self, momentum_term, iterator, gradients=[]):
        weight_moments = []
        weight_moments[iterator] = (momentum_term * self.weight_moments) + gradients

    def update_momentum(self, iterator, weights_moments=[]):
        return self.weights_matrices[iterator] - (self.alpha * self.weights_moments[iterator])
