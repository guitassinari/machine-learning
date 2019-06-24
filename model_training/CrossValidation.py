# coding: utf-8

from model_training.StratifiedDivisor import StratifiedDivisor
from performance.ModelPerformance import ModelPerformance
import numpy as np


class CrossValidation:
    def __init__(self, hyper_parameters, model_class, n_divisions, dataset):
        self.hyper_parameters_options = hyper_parameters
        self.model_klass = model_class
        self.divisor = StratifiedDivisor(dataset, n_divisions)
        self.divisions = n_divisions

    def get_performance_indexes(self):
        """
        Calcula o indice de performance para cada hiper parametro.
        :return: Lista com os indices de performance de cada hiper parametro
        """
        performances = []
        for parameters_index in range(len(self.hyper_parameters_options)):
            hyper_parameters = self.__hyper_parameters_at(parameters_index)
            performance = self.__calculate_performance_for(hyper_parameters)
            performances.append(performance)
        return performances

    def get_best_hyper_parameter(self):
        """
        retorna o melhor hiper parametro baseado nos indices de performance.
        Como os indicies representam a distancia do f measure para  1 (valor ideal),
        quanto menor o indice, melhor o resultado
        :return: Hiper parametro que obteve o melhor(menor) indice de performance
        """
        performances = self.get_performance_indexes()
        best_performance_index = performances.index(np.min(performances))
        return self.__hyper_parameters_at(best_performance_index)

    def __calculate_performance_for(self, hyper_parameters):
        """
        Calcula a performance para um dado hiper parametro através de Cross Validation.
        Para cada possível divisão do dataset, um modelo floresta é treinado e validado
        com o conjunto de teste da divisão. Com essa validação é calculado o f1 measure,
        que deve ser exatamente 1 para um modelo perfeito.
        Retornamos a média das distâncias de cada f1 measure para o valor ideal: 1
        :param hyper_parameters: hiper parametro para avaliar
        :return: média das distâncias de F1 Score para 1.
        """
        individual_performances = []
        for division_version in range(self.divisions):
            model = self.__create_model(hyper_parameters, division_version)
            performance_index = self.__model_performance_index(model, division_version)
            individual_performances.append(performance_index)
        return np.absolute(1 - np.mean(individual_performances))

    def __hyper_parameters_at(self, i):
        """
        :param i: indice do hiper parametro desejado
        :return: hiper parametro desejado
        """
        return self.hyper_parameters_options[i]

    def __create_model(self, hyper_parameters, division_version):
        """
        Cria um modelo com dados de treinamento de cross dado o hiper parametro
        validation, dada a versão da divisão
        :param hyper_parameters: hiper parametros do modelo
        :param division_version: indice da divisão de cross validation
        :return: um modelo treinado
        """
        training_set = self.divisor.get_training_set(division_version)
        return self.model_klass(hyper_parameters, training_set)

    def __model_performance_index(self, model, division_version):
        """
        Calcula o indice de performance de um modelo. No caso, F1 Score.
        :param model: modelo a ser testado
        :param division_version: versão do dataset de teste de cross validation a ser utilizado
        :return: o indice F1 Score do modelo
        """
        test_set = self.divisor.get_test_set(division_version)
        return ModelPerformance(model, test_set).f1_measure()

