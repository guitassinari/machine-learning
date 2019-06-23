# coding: utf-8


from data.DatasetFile import DatasetFile
from data.NetworkStructure import NetworkStructure
from data.InitialWeights import InitialWeights
from models.NeuralNetworkMath import NeuralNetworkMath
import copy
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-cp", dest="class_position",
                    help="class position inside dataset file, like an array index",
                    default=-1,
                    nargs=1,
                    type=int)
parser.add_argument("-cn", dest="normalization",
                    help="1 normalize, zero do not.",
                    default=1,
                    nargs=1,
                    type=int)

parser.add_argument("-model", dest="model_name",
                    help="Model to be used. Can be either Forest or NeuralNetwork",
                    nargs=1, default="Forest")
parser.add_argument("network",
                    help="Defines the neural network structure file path")
parser.add_argument("weights",
                    help="Neural network starting weights file path")
parser.add_argument("dataset", help="dataset file path")



def tab(num_tab):
    return " " * num_tab

def run():
    args = parser.parse_args()
    dataset_file_path = args.dataset
    network_file_path = args.network
    weights_file_path = args.weights
    class_position = args.class_position[0]
    norm = args.normalization

    print("\n")
    print("Dataset path:", dataset_file_path)
    print("Network Structure path:", network_file_path)
    print("Initial weights path:", weights_file_path)
    print("\n")

    print("------------ LENDO LAMBDA E ESTRUTURA DA REDE --------------\n")
    layers, _lambda = NetworkStructure(network_file_path).read()
    print("LAMBDA", _lambda)
    print("LAYERS", layers)
    print("\n")

    print("------------ LENDO PESOS E BIASES --------------\n")
    weights, bias = InitialWeights(weights_file_path).read()
    print("WEIGHTS", weights)
    print("BIASES", bias)
    print("\n")

    print("------------ LENDO DATASET --------------\n")
    dataset = DatasetFile(dataset_file_path, class_position, norm).read()

    examples = dataset.get_examples()

    print("------------ CALCULANDO ERRO / CUSTO -----------------\n")


    for example_index in range(len(examples)):
        example = examples[example_index]
        float_input = list(map(lambda string_attr: float(string_attr), example.get_body()))
        print("Exemplo", example_index + 1)
        print(tab(2), "x: ", float_input)
        print(tab(2), "y: ", example.get_class())
        expected_output = NeuralNetworkMath.example_expected_output(example, dataset)
        print(tab(2), "Expected y:", expected_output)
        all_activations = NeuralNetworkMath.all_activations_for(inputs=[float_input],
                                                                weights_matrices=weights,
                                                                bias_matrices=bias)
        real_output = all_activations[-1]
        print(tab(2), "Got y:", real_output)

        print("")
        print("ACTIVATIONS:")
        for activation in all_activations:
            print(tab(2), activation)

        print("")
        cost = NeuralNetworkMath.loss(real_output, expected_output)
        print("COST:")
        print(tab(2), cost)
        print("")
        print("NUMERICAL VERIFICATION")
        numerical_grad_matrix = NeuralNetworkMath.numerical_verifier(float_input, weights, bias, 1e-07, expected_output)

        print(tab(2), "Gradient Verification...")
        print("For example", example_index+1)
        for index in range(len(numerical_grad_matrix)):
            print(tab(4), "Theta", index+1)
            print(tab(6), numerical_grad_matrix[index])
        print("----------------")

    print("------------ BACKPROPAGATING ------------------------\n")

    all_gradients = []
    for example_index in range(len(examples)):
        example = examples[example_index]
        float_input = list(map(lambda string_attr: float(string_attr), example.get_body()))
        expected_output = NeuralNetworkMath.example_expected_output(example, dataset)
        all_activations = NeuralNetworkMath.all_activations_for(inputs=[float_input],
                                                                weights_matrices=weights,
                                                                bias_matrices=bias)

        deltas, bias_deltas = NeuralNetworkMath.calculate_deltas(weights_matrices=weights,
                                                                 expected_outputs=expected_output,
                                                                 activations=all_activations)

        gradients = NeuralNetworkMath.all_gradients(activations_matrices=all_activations,
                                                    deltas_matrices=deltas)
        all_gradients.append(gradients)

        print("Exemplo", example_index+1)

        for layer in range(len(deltas)):
            print(tab(2), "Delta", layer+2)
            layer_deltas = deltas[layer]
            for neuron_delta in layer_deltas:
                print(tab(4), neuron_delta)

        print("\n")

        print(tab(2), "Gradients (without bias)")
        for index in range(len(gradients)):
            print(tab(4), "Theta", index+1)

            theta_gradient = gradients[index]
            for gradient_matrix in theta_gradient:
                print(tab(6), gradient_matrix)

        print("\n")


    gradient_regularization = NeuralNetworkMath.gradients_regularization(
        weights_matrices=weights,
        _lambda=_lambda)

    gradient_sum = []
    for index in range(len(all_gradients)):
        if index == 0:
            gradient_sum = all_gradients[0]
        else:
            gradient_sum = NeuralNetworkMath.matrix_list_operation(
                np.add,
                gradient_sum,
                all_gradients[index]
            )

    sum_with_regularization = NeuralNetworkMath.matrix_list_operation(
        np.add,
        gradient_sum,
        gradient_regularization
    )

    regularized_gradients = list(map(lambda matrix: np.divide(matrix, len(examples)), sum_with_regularization))

    print("Regularized Gradients")
    for index in range(len(regularized_gradients)):
        print(tab(4), "Theta", index + 1)

        theta_gradient = regularized_gradients[index]
        for gradient_matrix in theta_gradient:
            print(tab(6), gradient_matrix)

run()
