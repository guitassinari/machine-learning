from ConfusionMatrix import ConfusionMatrix
from ConfusionMatrixToConfusionTable import ConfusionMatrixToConfusionTable
import numpy as np


class ModelPerformance:
    BETA = 1

    def __init__(self, model, test_set):
        self.confusion_matrix = ConfusionMatrix(model, test_set)
        self.matrix_to_table_parser = ConfusionMatrixToConfusionTable(self.confusion_matrix)

    def f1_measure(self):
        f1s = []
        for klass in self.__matrix_classes():
            f1s.append(self.__confusion_table_for(klass).f_score())
        return np.mean(f1s)

    def __confusion_table_for(self, klass):
        return self.matrix_to_table_parser.confusion_table_for(klass)

    def __matrix_classes(self):
        return self.confusion_matrix.possible_classes()
