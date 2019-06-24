# coding: utf-8


from model_training.CrossValidation import CrossValidation
from charts.LineChart import LineChart
from data.DatasetFile import DatasetFile
from data.HyperParametersFile import HyperParametersFile
import numpy as np
import argparse
import importlib

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("dataset", help="dataset file path")
parser.add_argument("parameters", help="hyper parameters file path")
parser.add_argument("-cv", dest="cross_validation_folds",
                    help="number of cross validation folds",
                    default=[2],
                    nargs=1,
                    type=int)
parser.add_argument("-cp", dest="class_position",
                    help="class position inside dataset file, like an array index",
                    default=[-1],
                    nargs=1,
                    type=int)
parser.add_argument("-cn", dest="normalization",
                    help="1 normalize, zero do not.",
                    default=[1],
                    nargs=1,
                    type=int)
parser.add_argument("-model", dest="model_name",
                    help="Model to be used. Can be either Forest or NeuralNetwork",
                    type=str, default="NeuralNetwork")
parser.add_argument("-ignore", dest="ignore_columns",
                    help="indexes of example attributes to be ignored when reading dataset",
                    nargs="+",
                    type=int,
                    default=[])


def run():
    args = parser.parse_args()
    dataset_file_path = args.dataset
    hyper_parameters_file_path = args.parameters
    cv_divisions = args.cross_validation_folds[0]
    class_position = args.class_position[0]
    ignore_columns = args.ignore_columns
    norm = args.normalization[0] == 1

    model_name = args.model_name
    module = importlib.import_module("models."+model_name)
    model_class = getattr(module, model_name)

    print("\n")
    print("Dataset path:", dataset_file_path)
    print("Hyper parameters path:", hyper_parameters_file_path)
    print("Cross Validation K-Fold:", cv_divisions)
    print("Class position: ", class_position)
    print("\n")
    dataset = DatasetFile(dataset_file_path, class_position, norm, ignore_columns).read()

    hyper_parameters_list = HyperParametersFile(hyper_parameters_file_path).read()
    cv = CrossValidation(hyper_parameters_list, model_class, cv_divisions, dataset)

    performance_indexes = cv.get_performance_indexes()
    best_hyper_parameter_index = performance_indexes.index(np.max(performance_indexes))
    best_hyper_parameter = hyper_parameters_list[best_hyper_parameter_index]
    print(best_hyper_parameter)

    layer_structures = list(map(lambda hp: hp["layers_structure"], hyper_parameters_list))
    LineChart([performance_indexes], layer_structures, min=0, max=1)
    LineChart.show_charts()


run()
