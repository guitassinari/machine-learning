from models.DecisionTree import DecisionTree


class Forest:
    def __init__(self, hyper_parameters, training_set):
        """
        Inicializa a floresta com suas árvores.

        :param hyper_parameters: dictionary/hash contendo os hiper parâmetros
        :param training_set: dataset de treinamento
        """
        self.number_of_trees = hyper_parameters["n_trees"]
        self.trees = []
        self.training_set = training_set
        sample_size = round(2*training_set.size() / 3)

        # Cria todas as number_of_trees árvores de decisão
        for i in range(self.number_of_trees):
            # resampling usando bootstrap stratificado
            tree_training_set = self.training_set.resample(sample_size)
            tree = DecisionTree(hyper_parameters, tree_training_set)
            self.trees.append(tree)

    def predict(self, example):
        """
        Pede que todas as árvores façam uma predição para o exemplo e retorna
        o valor mais retornado / frequente [votação]
        :param example: instância na forma de um Example para a qual se quer prever a classe
        :return: classe predita para o example
        """
        predictions = self.__trees_predictions_for(example)
        max_frequency_so_far = 0
        major = predictions[0]
        for klass in predictions:
            klass_frequency = predictions.count(klass)
            if klass_frequency > max_frequency_so_far:
                max_frequency_so_far = klass_frequency
                major = klass
        print(predictions, major)
        return major

    def __trees_predictions_for(self, example):
        return list(map(lambda tree: tree.predict(example), self.trees))
