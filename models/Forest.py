import random
from models.DecisionTree import DecisionTree
from sklearn.utils import resample
from data.Dataset import Dataset


class Forest:
    def __init__(self, hyper_parameters, training_set):
        """
        Inicializa a floresta com suas árvores.

        :param hyper_parameters: dictionary/hash contendo os hiper parâmetros
        :param training_set: dataset de treinamento
        """
        self.number_of_trees = hyper_parameters["n_trees"]
        self.attr_sample_num = hyper_parameters["n_attr_sample"]
        self.trees = []
        self.training_set = training_set
        classes = training_set.get_classes()
        examples = training_set.get_examples()
        sample_size = round(training_set.size() / 3)

        # Cria todas as number_of_trees árvores de decisão
        for i in range(self.number_of_trees):
            # resampling usando bootstrap stratificado
            sample_examples = resample(examples,
                                       n_samples=sample_size,
                                       stratify=classes)
            tree_training_set = Dataset(sample_examples)
            tree = DecisionTree(hyper_parameters, tree_training_set)
            self.trees.append(tree)

    def predict(self, example):
        """
        TODO: realizar um predict em todas as árvores e retornar por votação de maioria

        :param example: instância na forma de um Example para a qual se quer prever a classe
        :return: classe predita para o example
        """
        return random.choice(self.training_set.get_uniq_classes())
