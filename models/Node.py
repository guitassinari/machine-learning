import numpy as np
import random
from models.InfoGain import InfoGain
from data.Attribute import AttributeType


class Node:
    MAX_DEPTH = 50
    """
    Essa classe repreenta um nodo da árvore de decisão
    """

    def __init__(self, hyper_parameters, dataset, depth=1):
        self.depth = depth
        self.hyper_parameters = hyper_parameters
        self.n_attr_sample = hyper_parameters["n_attr_sample"]
        self.dataset = dataset
        self.attr_names = dataset.get_attr_names()
        self.splitter = None
        self.attribute = None
        if (not (dataset.empty() or dataset.pure())) and depth <= self.MAX_DEPTH:
            self.attribute = self.__best_attribute()
            self.splitter = self.__split()

    def predict(self, example):
        # Se dataset vazio, retorna None
        if self.dataset.empty():
            return None

        # Se só tiver uma classe no seu dataset, retorna essa classe
        if self.dataset.pure():
            return self.dataset.get_example_at(0).get_class()

        if self.splitter is None:
            return self.dataset.major_class()

        # Manda a predição pros nodes filhos
        prediction = self.splitter.predict(example)

        # Se a predição de nodes filhos é nula, retorna a classe mais frequente
        # no dataset
        if prediction is not None:
            return prediction

        return self.dataset.major_class()

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
        return SplitterClass(self.hyper_parameters, self.dataset, self.attribute, self.depth)

    def __best_attribute(self):
        """
        Seleciona o melhor atributo para o node atual, levando em consideração o
        info gain máximo
        :return: nome do melhor atributo para este node
        """
        attributes = self.__attributes_sample()
        attributes_info_gain = list(map(
            lambda attr: InfoGain(attr, self.dataset).value(),
            attributes
        ))
        index_of_max_info_gain = attributes_info_gain.index(np.max(attributes_info_gain))
        return attributes[index_of_max_info_gain]

    def __attributes_sample(self):
        """
        pega uma amostra de tamanho n_attr_sample dos atributos possíveis
        :return: uma lista de atributos
        """
        return random.sample(self.attr_names, self.n_attr_sample)


class NodeSplitStrategy:
    """
    A responsabilidade dessa classe é, dado um tipo de atributo, retornar
    o divisor correto [catégorico ou numérico]
    """
    @classmethod
    def for_type(cls, attr_type):
        if attr_type == AttributeType.CATEGORIC:
            return CategoricSplitter
        else:
            return NumericSplitter


class NumericSplitter:
    """
    Divisor de node para atributos numéricos
    """
    def __init__(self, hyper_parameters, dataset, attr_name, depth):
        self.depth = depth
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
        if self.__attr_value_for(example) <= self.divider:
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
        first_dataset_indexes = []
        second_dataset_indexes = []
        examples = self.dataset.get_examples()
        for i in range(len(examples)):
            example = examples[i]
            if self.__attr_value_for(example) <= divider:
                first_dataset_indexes.append(i)
            else:
                second_dataset_indexes.append(i)
        return [
            Node(self.hyper_parameters, self.dataset.subset(first_dataset_indexes), self.depth+1),
            Node(self.hyper_parameters, self.dataset.subset(second_dataset_indexes), self.depth+1)
        ]

    def __mean(self):
        """
        :return: média dos valores do attributo no dataset
        """
        attr_values = self.dataset.get_attr_value(self.attr_name)
        return np.mean(list(map(lambda value: float(value), attr_values)))

    def __attr_value_for(self, example):
        return float(example.get_attr_value(self.attr_name))


class CategoricSplitter:
    def __init__(self, hyper_parameters, dataset, attr_name, depth):
        self.depth = depth
        self.hyper_parameters = hyper_parameters
        self.dataset = dataset
        self.attr_name = attr_name
        self.possible_values = dataset.get_uniq_attr_values(attr_name)
        self.nodes = self.__get_nodes()

    def predict(self, example):
        example_attr_value = example.get_attr_value(self.attr_name)
        node = self.nodes[self.possible_values.index(example_attr_value)]
        return node.predict(example)

    def __get_nodes(self):
        nodes = []
        if self.attr_name == 3:
            print(self.possible_values)
        for value in self.possible_values:
            split_dataset = self.__split_dataset_for(value)
            if self.attr_name == 3:
                print(value, split_dataset.get_bodies())
            nodes.append(Node(self.hyper_parameters, split_dataset, self.depth+1))
        return nodes

    def __split_dataset_for(self, attr_value):
        return self.dataset.split_dataset_for(self.attr_name, attr_value)
