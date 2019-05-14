# coding: utf-8

from data.DatasetFile import DatasetFile
from models.Node import Node
import copy

names = ["Tempo", "Temperatura", "Umidade", "Ventoso", "Joga"]


def print_tree(node, level=0):
    if node.attribute is None:
        return
    string = (" " * level*4) + names[node.attribute]
    print(string)
    if node.splitter:
        for _node in node.splitter.nodes:
            print_tree(_node, level+1)

dataset = DatasetFile("test_benchmark.csv").read()
print(dataset.get_attr_names())
node = Node({"n_attr_sample": 3}, dataset)
example = copy.deepcopy(dataset.get_example_at(0))
print(node.predict(example))

print_tree(node)
