import numpy as np
import random
from models.InfoGain import InfoGain
from data.Attribute import AttributeType
from data.Dataset import Dataset


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


class NodeSplitStrategy:
    """
    A responsabilidade dessa classe é, dado um tipo de atributo, retornar
    o divisor correto [catégorico ou numérico]
    """
    @classmethod
    def for_type(cls, attr_type):
        if attr_type == AttributeType.CATEGORIC:
            return NumericSplitter
        else:
            return CategoricSplitter


class NumericSplitter:
    """
    Divisor de node para atributos numéricos
    """
    def __init__(self, hyper_parameters, dataset, attr_name):
        self.hyper_parameters = hyper_parameters
        self.dataset = dataset
        self.attr_name = attr_name
        # o número divisor é a média entre todos os valores do dataset
        self.divider = self.__mean()
        self.nodes = self.__get_nodes()

    def predict(self, example):
        """
        Testa qual node representa o valor do exemplo e retorna a predição dele

        :param example: exemplo a ser predito
        :return: classe predita para o exemplo
        """
        # se o valor do exemplo é <= ao divisor, retorna a predição do primeiro node
        # senão retorna a predição do segundo
        if example.get_attr_value(self.attr_name) <= self.divider:
            return self.nodes[0].predict(example)
        else:
            return self.nodes[1].predict(example)

    def __get_nodes(self):
        """
        Divide o dataset atual em 2:
            Um com valores <= à média
            Outro com valores > à média
        :return: lista com dois nodes.
        """
        divider = self.__mean()
        first_dataset = []
        second_dataset = []
        for example in self.dataset.get_examples():
            if example.get_attr_value(self.attr_name) <= divider:
                first_dataset.append(example)
            else:
                second_dataset.append(example)
        return [
            Node(self.hyper_parameters, Dataset(first_dataset)),
            Node(self.hyper_parameters, Dataset(second_dataset))
        ]

    def __mean(self):
        """

        :return: média dos valores do attributo no dataset
        """
        attr_values = self.dataset.get_attr_value(self.attr_name)
        return np.mean(attr_values)


class CategoricSplitter:
    def __init__(self, hyper_parameters, dataset, attr_name):
        self.hyper_parameters = hyper_parameters
        self.dataset = dataset
        self.attr_name = attr_name
        self.possible_values = dataset.get_uniq_attr_values(attr_name)
        self.nodes = self.__get_nodes()

    def predict(self, example):
        example_attr_value = example.get_attr_value(self.attr_name)
        node = self.nodes.index(self.possible_values[example_attr_value])
        return node.predict(example)

    def get_nodes(self):
        nodes = []
        for value in self.possible_values:
            split_dataset = self.__split_dataset_for(value)
            nodes.append(Node(self.hyper_parameters, split_dataset))
        return nodes

    def __split_dataset_for(self, attr_value):
        new_dataset = []
        for example in self.dataset.get_examples():
            if example.get_attr_value(self.attr_name) == attr_value:
                new_dataset.append(example)
        return Dataset(new_dataset)
