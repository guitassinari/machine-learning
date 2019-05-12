from data.Dataset import Dataset
from data.Example import Example
from models.Node import Node

dataset = Dataset([
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["ensolarado", "quente", "alta", "falso", "nao"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["ensolarado", "quente", "alta", "falso", "nao"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["nublado", "quente", "alta", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["chuvoso", "amena", "alta", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["chuvoso", "fria", "normal", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["chuvoso", "fria", "normal", "verdadeiro", "nao"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["nublado", "amena", "alta", "falso", "nao"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["ensolarado", "fria", "normal", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["ensolarado", "fria", "normal", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["chuvoso", "amena", "normal", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["ensolarado", "amena", "normal", "verdadeiro", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["nublado", "amena", "alta", "", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["nublado", "quente", "normal", "falso", "sim"]),
    Example(["TEMPO", "TEMPERATURA", "UMIDADE", "VENTOSO", "JOGA"], ["chuvoso", "amena", "alta", "verdadeiro", "nao"]),
])

    # 1 - seleciona atributos randomicamente (n = sqrt(n))
    # calcula o ganho de informação entre esses attributos
    # seleciona o com maior ganho de informacao para ser a raiz da subarvore
    # se for numerico, calcula a media entre todos possiveis
    # se for categorico, cria nodos para todos os possiveis
    # repita para todos os nodos criados do passo 1 até aqui


class DecisionTree:
    def __init__(self, hyper_parameters, training_set):
        self.training_set = training_set
        self.hyper_parameters = hyper_parameters
        self.attr_sample_num = hyper_parameters["n_attr_sample"]
        self.root = self.__create_root_node()

    def predict(self, example):
        return self.root.predict(example)

    def __create_root_node(self):
        return Node(self.hyper_parameters, self.training_set)

