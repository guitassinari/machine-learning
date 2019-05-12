# coding: utf-8

"""
    Dividir o dataset em partes e, a partir dessas partes, selecionar
"""


from sklearn.model_selection import StratifiedKFold
import numpy as np
from data.Dataset import Dataset


class StratifiedDivisor:
    def __init__(self, dataset, divisions):
        self.bodies = dataset.get_bodies()
        self.classes = dataset.get_classes()
        self.dataset = dataset
        self.training_sets_indexes = []
        self.test_sets_indexes = []
        parts_generator = StratifiedKFold(n_splits=divisions)\
            .split(self.bodies, self.classes)

        for train, test in parts_generator:
            self.training_sets_indexes.append(train)
            self.test_sets_indexes.append(test)

    def get_training_set(self, version):
        examples_indexes = self.training_sets_indexes[version]
        return Dataset(self.__get_examples(examples_indexes))

    def get_test_set(self, version):
        examples_indexes = self.test_sets_indexes[version]
        return Dataset(self.__get_examples(examples_indexes))

    def __get_examples(self, set_indexes):
        # transforma em array numpy pra poder pegar exemplos com uma lista de indices
        np_array_examples = np.array(self.dataset.get_examples())
        filtered_examples = np_array_examples[set_indexes]
        return filtered_examples.tolist()
