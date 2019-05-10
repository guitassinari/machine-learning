# coding: utf-8

# Dividir o dataset em partes e, a partir dessas partes, selecionar
# training set
# validation set
# e test set

from Dataset import Dataset
import pandas as pd

class StratifiedDivisor:
    def __init__(self, dataset, divisions):
        # se divisions > exemplos no dataset, retornar erro
        pass

    def get_training_set(self, version):
        # se a versão for maior que o maximo possível de variações,
        # considerar como se a contagem reinicia-se
        # Lembrando que o algoritmo deve ser estratificado, ou seja,
        # os sets divididos devem ter a mesma porcentagem de classes observadas
        return Dataset()

    def get_test_set(self, version):
        return Dataset()

    def get_validation_set(self, version):
        return Dataset()
