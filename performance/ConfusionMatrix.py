"""
    Uma matriz de confusão. Não confundir com uma tabela de confusão.

    A matrix de confusão possui mais do que uma duas linhas e duas colunas,
    por isso difere da tabela de confusão, que possui duas linhas e duas colunas

    Para criar a matriz de confusão escolhi o formato de dictionary da seguinte maneira:

    O dict

    - O primeiro nível do dictionary uma linha da matriz de confusão.

"""

class ConfusionMatrix:
    BETA = 1

    def __init__(self, model, test_set):
        confusion_hash = {}
        possible_classes = test_set.get_uniq_classes()

        # {
        #   "sim": { "sim": 3, "nao": 2 }, quando previu sim, 3 realmente eram sims, dois deveriam ser naos
        #   "nao": { "sim": 2, "nao": 1 }
        # }
        for klass in possible_classes:
            confusion_hash[klass] = {}
            for klass_2 in possible_classes:
                confusion_hash[klass][klass_2] = 0

        for example in test_set.examples:
            correct_klass = example.get_class()
            predicted_klass = model.predict(example)
            confusion_hash[predicted_klass][correct_klass] += 1

        self.classes = possible_classes
        self.confusion_hash = confusion_hash

    def predictions_for(self, klass):
        return self.confusion_hash[klass].copy()

    def possible_classes(self):
        return self.classes.copy()
