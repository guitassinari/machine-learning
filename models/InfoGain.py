import math


class InfoGain:
    def __init__(self, hyper_parameters, dataset, attr_name):
        self.dataset = dataset
        self.attr_name = attr_name

    def value(self):
        dataset_entropy = Entropy(self.dataset)
        return dataset_entropy.total - dataset_entropy.for_attribute(self.attr_name)


class Entropy:
        def __init__(self, dataset):
            self.dataset = dataset

        def total(self):
            sum = 0
            for klass in dataset.get_uniq_classes():
                klass_prob = __probability_for(klass)
                sum += klass_prob * math.log2(klass_prob)
            return -sum

        def for_attribute(self, attr_name): # seria aqui? SIM!
            sum = 0
            possible_values = dataset.get_uniq_attr_values(attr_name)
            for value in possible_values:
                subset = dataset.split_dataset_for(attr_name, value)
                subset_entropy = Entropy(dataset).total()
                subset_proportion = subset.size() / self.dataset.size()
                sum += subset_proportion * subset_entropy
            return sum

        def __probability_for(self, klass):
            all_classes = dataset.classes()
            return all_classes.count(klass) / len(all_classes)
