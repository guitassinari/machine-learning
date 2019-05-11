import random

class Forest:
    TEST = ["a", "c", "d", "n", "s"]

    def __init__(self, hyper_parameters):
        self.number_of_trees = hyper_parameters["n_trees"]
        self.attr_sample_num = hyper_parameters["n_attr_sample"]

    def predict(self, example):
        # utilizar as árvores de decisão e, a partir das suas predições
        # utilizar o algoritmo de votação majoritária
        return random.choice(self.TEST)