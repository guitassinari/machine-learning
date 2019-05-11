# coding: utf-8

from model_training.StratifiedDivisor import StratifiedDivisor
from data.Example import Example
from data.Dataset import Dataset
from model_training.CrossValidation import CrossValidation
from models.Forest import Forest
from charts.LineChart import LineChart
from data.DatasetFile import DatasetFile
from data.HyperParametersFile import HyperParametersFile
import numpy as np
import sys

# dataset = Dataset([
#     Example(["nome", "velho"], ["Gui", "s"]),
#     Example(["nome", "velho"], ["Samuel", "n"]),
#     Example(["nome", "velho"], ["Fernando", "s"]),
#     Example(["nome", "velho"], ["Rafael", "n"]),
#     Example(["nome", "velho"], ["Rafael", "a"]),
#     Example(["nome", "velho"], ["Rafael", "a"]),
#     Example(["nome", "velho"], ["Rafael", "d"]),
#     Example(["nome", "velho"], ["Rafael", "c"]),
#     Example(["nome", "velho"], ["Rafael", "d"]),
#     Example(["nome", "velho"], ["Rafael", "d"]),
#     Example(["nome", "velho"], ["Rafael", "c"])
# ])
#
# divisor = StratifiedDivisor(dataset, 2)
#
# hyper_paremeters = [{"n_trees": 1, "n_attr_sample": 2},
#                     {"n_trees": 3, "n_attr_sample": 2}]
#
# cv = CrossValidation(hyper_paremeters, Forest, 2, dataset)
#
# print(cv.get_best_hyper_parameter())
#
# chart = LineChart([np.random.randn(1000)])
# LineChart.show_charts()
#


dataset_file_path = sys.argv[1]
hyper_parameters_file_path = sys.argv[2]
cv_divisions = int(sys.argv[3])

print("Dataset path:", dataset_file_path)
print("Hyper parameters path:", hyper_parameters_file_path)
print("Cross Validation K-Fold:", cv_divisions)


dataset = DatasetFile(dataset_file_path).read()
hyper_parameters_list = HyperParametersFile(hyper_parameters_file_path).read()

cv = CrossValidation(hyper_parameters_list, Forest, cv_divisions, dataset)

best_hyper_parameter = cv.get_best_hyper_parameter()
print(best_hyper_parameter)
print(cv.get_performance_indexes())

LineChart([cv.get_performance_indexes()])
LineChart.show_charts()


best_forest = Forest(best_hyper_parameter, dataset)
