from models.Node import Node


class DecisionTree:
    def __init__(self, hyper_parameters, training_set):
        self.training_set = training_set
        self.attr_sample_num = hyper_parameters["n_attr_sample"]
        self.root = self.__create_root_node()

    def predict(self, example):
        return self.root.predict(example)

    def __create_root_node(self):
        return Node(self.hyper_parameters, self.training_set)

