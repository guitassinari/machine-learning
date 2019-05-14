# coding: utf-8

from model_training.CrossValidation import CrossValidation
from models.Forest import Forest
from charts.LineChart import LineChart
from data.DatasetFile import DatasetFile
from data.HyperParametersFile import HyperParametersFile
import sys
import numpy as np
import cProfile


def run():
    dataset_file_path = sys.argv[1]
    hyper_parameters_file_path = sys.argv[2]
    cv_divisions = int(sys.argv[3])

    print("Dataset path:", dataset_file_path)
    print("Hyper parameters path:", hyper_parameters_file_path)
    print("Cross Validation K-Fold:", cv_divisions)

    dataset = DatasetFile(dataset_file_path).read()

    hyper_parameters_list = HyperParametersFile(hyper_parameters_file_path).read()
    cv = CrossValidation(hyper_parameters_list, Forest, cv_divisions, dataset)

    performance_indexes = cv.get_performance_indexes()
    best_hyper_parameter_index = performance_indexes.index(np.min(performance_indexes))
    best_hyper_parameter = hyper_parameters_list[best_hyper_parameter_index]
    print(best_hyper_parameter)

    LineChart([performance_indexes])
    LineChart.show_charts()

    # Forest(hyper_parameters_list[0], dataset)

# cProfile.run("run()")

run()
