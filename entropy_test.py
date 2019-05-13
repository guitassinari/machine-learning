# coding: utf-8


from data.DatasetFile import DatasetFile
from models.InfoGain import Entropy



dataset = DatasetFile("test_benchmark.csv").read()
entropy = Entropy(dataset)
print(dataset.get_attr_names())
print(entropy.total() == 0.9402859586706311)
print(entropy.for_attribute(0), 0.6935361388961918)
