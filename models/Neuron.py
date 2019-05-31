import math


class Neuron:
    def __init__(self, initial_weights = []):
        self.weights = initial_weights
        pass

    def predict(self, features):
        """
        :param features: uma lista de valores numéricos (deve ser do mesmo
        tamanho de self.weights)
        :return:
        """
        multiplications = [weight*feature for weight, feature in zip(self.weights, features)]
        return self.sigmoid(sum(multiplications))
        pass

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
