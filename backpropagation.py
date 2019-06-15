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

    print("--------------------- Calculando Erro -----------------\n")
    for example_index in range(len(examples)):
        example = examples[example_index]
        float_input = list(map(lambda string_attr: float(string_attr), example.get_body()))
        print("Exemplo", example_index)
        print("x: ", float_input)
        print("y: ", example.get_class())
        expected_output = NeuralNetworkMath.example_expected_output(example, dataset)
        print("Expected y:", expected_output)
        all_activations = NeuralNetworkMath.all_activations_for(inputs=[float_input],
                                                                weights_matrices=weights,
                                                                bias_matrices=bias)
        real_output = all_activations[-1]
        print("Got y:", real_output)

        print("")
        print("ACTIVATIONS:")
        for activation in all_activations:
            print(activation)

        print("")
        cost = NeuralNetworkMath.loss(real_output, expected_output)
        print("COST: ", cost)
        print("")







run()
