import math


class InfoGain:
    def __init__(self, attr_name, dataset):
        self.dataset = dataset
        self.attr_name = attr_name

    def value(self):
        """
        :return: Ganho de informação do atributo self.attr_name para o self.dataset
        """
        dataset_entropy = Entropy(self.dataset)
        gain = dataset_entropy.total() - dataset_entropy.for_attribute(self.attr_name)
        print(self.attr_name, gain)
        return round(gain, 4)


class Entropy:
    def __init__(self, dataset):
        self.dataset = dataset

    def total(self):
        """
        :return: Entropia total do dataset [Info(D)]
        """
        _sum = 0
        for klass in self.dataset.get_uniq_classes():
            klass_prob = self.__probability_for(klass)
            if klass_prob == 0:
                continue
            _sum += klass_prob * math.log2(klass_prob)
        return -_sum

    def for_attribute(self, attr_name):
        """
        :param attr_name: atributo para calcular sua entropia
        :return: entropia do dataset para o atributo attr_name [InfoA(D)]
        """
        _sum = 0
        possible_values = self.dataset.get_uniq_attr_values(attr_name)
        for value in possible_values:
            subset = self.dataset.split_dataset_for(attr_name, value)
            if subset.empty():
                continue
            subset_entropy = Entropy(subset).total()
            subset_proportion = subset.size() / self.dataset.size()
            _sum += subset_proportion * subset_entropy
        return _sum

    def __probability_for(self, klass):
        """
        probablidade da classe no dataset atual n_classe/total_classes
        :param klass: classe cuja probabilidade se deseja
        :return: probabilidade de a classe acontecer no dataset
        """
        all_classes = self.dataset.get_classes()
        return all_classes.count(klass) / len(all_classes)
