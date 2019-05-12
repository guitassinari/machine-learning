from data.Dataset import Dataset
import numpy as np
from data.Attribute import AttributeType
from models.Node import Node


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
