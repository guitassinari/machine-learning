import numpy as np
import random
from models.NodeSplitStrategy import NodeSplitStrategy
from models.InfoGain import InfoGain


class Node:
    """
    Essa classe repreenta um nodo da árvore de decisão
    """

    def __init__(self, hyper_parameters, dataset):
        self.hyper_parameters = hyper_parameters
        self.n_attr_sample = hyper_parameters["n_attr_sample"]
        self.dataset = dataset
        self.attribute = self.__best_attribute()
        self.children_nodes = []
        if not (self.pure() or self.empty()):
            self.splitter = self.__split()

    def predict(self, example):
        # Se só tiver uma classe no seu dataset, retorna essa classe
        if self.pure():
            return self.dataset.get_classes()[0]

        # Se dataset vazio, retorna None
        if self.empty():
            return None

        # Manda a predição pros nodes filhos
        prediction = self.splitter.predict(example)

        # Se a predição de nodes filhos é nula, retorna a classe mais frequente
        # no dataset
        if prediction is None:
            return self.dataset.major_class()

    def pure(self):
        """
        Verifica se o dataset é puro. Ou seja, só possui uma classe nos seus exemplos

        :return: true se o dataset possui apenas uma classe em todos os seus exemplos
        """
        return len(self.dataset.get_uniq_attr_values()) == 1

    def empty(self):
        """
        Verifica se  o dataset está vazio
        :return: retorna true se o dataset não possui exemplos
        """
        return self.dataset.size() == 0

    def __split(self):
        """
        Cria um splitter (basicamente os nodes filhos e a lógica de predição
        para eles
        :return: um Splitter, catégorico ou numérico, que contém os nodes filhos
        e a lógica de predição
        """
        return self.__create_splitter()

    def __create_splitter(self):
        attr_type = self.dataset.get_attr_type(self.attribute)
        SplitterClass = NodeSplitStrategy.for_type(attr_type)
        return SplitterClass(self.hyper_parameters, self.dataset, self.attribute)

    def __best_attribute(self):
        """
        Seleciona o melhor atributo para o node atual, levando em consideração o
        info gain máximo
        :return: nome do melhor atributo para este node
        """
        attributes = self.__attributes_sample()
        attributes_info_gain = map(
            lambda attr: InfoGain(attr, self.dataset).value(),
            attributes
        )
        return attributes.index(np.max(attributes_info_gain))

    def __attributes_sample(self):
        """
        pega uma amostra de tamanho n_attr_sample dos atributos possíveis
        :return: uma lista de atributos
        """
        all_attributes = self.dataset.get_attr_names()
        return random.sample(all_attributes, self.n_attr_sample)
