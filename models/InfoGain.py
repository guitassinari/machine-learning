import math


class InfoGain:
    def __init__(self, attr_name, dataset):
        self.dataset = dataset
        self.attr_name = attr_name

    def value(self):
        dataset_entropy = Entropy(self.dataset)
        return dataset_entropy.total() - dataset_entropy.for_attribute(self.attr_name)


class Entropy:
    def __init__(self, dataset):
        self.dataset = dataset

    def total(self):
        _sum = 0
        for klass in self.dataset.get_uniq_classes():
            klass_prob = self.__probability_for(klass)
            _sum += klass_prob * math.log2(klass_prob)
        return -_sum

    def for_attribute(self, attr_name):
        _sum = 0
        possible_values = self.dataset.get_uniq_attr_values(attr_name)
        for value in possible_values:
            subset = self.dataset.split_dataset_for(attr_name, value)
            subset_entropy = Entropy(self.dataset).total()
            subset_proportion = subset.size() / self.dataset.size()
            _sum += subset_proportion * subset_entropy
        return _sum

    def __probability_for(self, klass):
        all_classes = self.dataset.get_classes()
        return all_classes.count(klass) / len(all_classes)
