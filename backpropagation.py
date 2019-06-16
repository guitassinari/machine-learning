# coding: utf-8


from data.DatasetFile import DatasetFile
from data.NetworkStructure import NetworkStructure
from data.InitialWeights import InitialWeights
from models.NeuralNetwork import NeuralNetwork
from models.NeuralNetworkMath import NeuralNetworkMath

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-cp", dest="class_position",
                    help="class position inside dataset file, like an array index",
                    default=-1,
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
    dataset = DatasetFile(dataset_file_path, class_position).read()
    examples = dataset.get_examples()

    print("------------ CALCULANDO ERRO / CUSTO -----------------\n")
    for example_index in range(len(examples)):
        example = examples[example_index]
        float_input = list(map(lambda string_attr: float(string_attr), example.get_body()))
        print("Exemplo", example_index)
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

    print("------------ BACKPROPAGATING ------------------------\n")

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

        print("Exemplo", example_index)
        print(tab(2), "Deltas")
        print(tab(4), deltas)

        print(tab(2), "Deltas Bias")
        print(tab(4), bias_deltas)

        print(tab(2), "Gradients")
        print(tab(4), gradients)
        print("\n")


run()
