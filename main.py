# coding: utf-8

from model_training.StratifiedDivisor import StratifiedDivisor
from data.Example import Example
from data.Dataset import Dataset
from model_training.CrossValidation import CrossValidation
from models.Forest import Forest

dataset = Dataset([
    Example(["nome", "velho"], ["Gui", "s"]),
    Example(["nome", "velho"], ["Samuel", "n"]),
    Example(["nome", "velho"], ["Fernando", "s"]),
    Example(["nome", "velho"], ["Rafael", "n"]),
    Example(["nome", "velho"], ["Rafael", "a"]),
    Example(["nome", "velho"], ["Rafael", "a"]),
    Example(["nome", "velho"], ["Rafael", "d"]),
    Example(["nome", "velho"], ["Rafael", "c"]),
    Example(["nome", "velho"], ["Rafael", "d"]),
    Example(["nome", "velho"], ["Rafael", "d"]),
    Example(["nome", "velho"], ["Rafael", "c"])
])

divisor = StratifiedDivisor(dataset, 2)

hyper_paremeters = [{"n_trees": 1, "n_attr_sample": 2},
                    {"n_trees": 3, "n_attr_sample": 2}]

cv = CrossValidation(hyper_paremeters, Forest, 2, dataset)

print(cv.get_best_hyper_parameter())
# dataset = DatasetFile("dadosBenchmark_validacaoAlgoritmoAD.csv").read()
#
# divisions = 5
# hyper_parameters_options = [
#     {
#         'number_of_trees': 5,
#         'number_of_attr_samples': 5
#     }
# ]
#
# print(f'Dataset será dividido em {divisions} partes, das quais uma para teste e uma para validação')
#
# dataset = DatasetFile("path").read
# divisor = StratifiedDivisor(dataset, divisions)
#
# forests = []
# for hyper_parameter in hyper_parameters_options:
#     forests.append(Forest(hyper_parameter['number_of_trees'],
#                           hyper_parameter['number_of_attr_samples']
#                           ))
#
# performances = []
# for forest in forests:
#     performances.append(ModelPerformance())
#
# # para cada possibilidade de divisão do dataset original (Ver Cross Validation)
# for i in range(divisions):
#     training_set = divisor.get_training_set(i)
#     test_set = divisor.get_test_set(i)
#     validation_set = divisor.get_validation_set(i)
#
#     # calcular a performance de cada floresta (média e desvio padrão)
#     for forest_index in range(len(forests)):
#         forest = forests[forest_index]
#         forest.train_with(training_set)
#         performances[forest_index].evaluate(forest, validation_set)
#
# # é assim mesmo que vamos validar o melhor modelo?
# performance_indicators = map(lambda performance: performance.indicator, performances)
# best_performance_index = performance_indicators.index(min(performance_indicators))
# best_hyper_parameters = hyper_parameters_options[best_performance_index]
#
#
# # a partir daqui, treinar o melhor modelo (best_config) com training set e validation set
# #depois, testar a performance com test_set
#
# forest = Forest(best_hyper_parameters['number_of_trees'],
#                 best_hyper_parameters['number_of_attr_samples'])
#
# # treinar com training_set + validation_set
# # forest.train_with()