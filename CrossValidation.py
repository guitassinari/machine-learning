from StratifiedDivisor import StratifiedDivisor
from ModelPerformance import ModelPerformance
import numpy as np

class CrossValidation:
    def __init__(self, hyper_parameters, model_class, n_divisions, dataset):
        self.hyper_parameters_options = hyper_parameters
        self.model_klass = model_class
        self.divisor = StratifiedDivisor(dataset, n_divisions)
        self.divisions = n_divisions
        pass

    def get_best_hyper_parameter(self):
        performances = []
        for parameters_index in range(len(self.hyper_parameters_options)):
            hyper_parameters = self.__hyper_parameters_at(parameters_index)
            performance = self.__calculate_performance_for(hyper_parameters)
            performances.append(performance)
        best_performance_index = performances.index(np.max(performances))
        return self.__hyper_parameters_at(best_performance_index)

    def __calculate_performance_for(self, hyper_parameters, divisions):
        individual_performances = []
        for division_version in range(divisions):
            model = self.__create_model(hyper_parameters, division_version)
            performance_index = self.__model_performance_index(model, division_version)
            individual_performances.append(performance_index)
        return np.mean(individual_performances)

    def __hyper_parameters_at(self, i):
        return self.hyper_parameters_options[i]

    def __create_model(self, hyper_parameters, division_version):
        training_set = self.divisor.get_training_set(division_version)
        return self.model_klass(hyper_parameters, training_set)

    def __model_performance_index(self, model, division_version):
        test_set = self.divisor.get_test_set(division_version)
        return ModelPerformance(model, test_set).f1_measure

