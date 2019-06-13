# coding: utf-8


from data.DatasetFile import DatasetFile
from data.NetworkStructure import NetworkStructure
from data.InitialWeights import InitialWeights
from models.NeuralNetwork import NeuralNetwork
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
    class_position = args.class_position

    print("\n")
    print("Dataset path:", dataset_file_path)
    print("Network Structure path:", network_file_path)
    print("Initial weights path:", weights_file_path)
    print("\n")

    dataset = DatasetFile(dataset_file_path, class_position).read()
    network = NetworkStructure(network_file_path).read()
    weights = InitialWeights(weights_file_path).read()




run()
