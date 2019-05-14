# coding: utf-8

from data.DatasetFile import DatasetFile
from models.Node import Node
from data.Dataset import Dataset
from data.Example import Example
from performance.ConfusionMatrix import ConfusionMatrix
from performance.ConfusionMatrixToConfusionTable import ConfusionMatrixToConfusionTable
from performance.ModelPerformance import ModelPerformance
import copy
from sklearn.metrics import f1_score

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

test_dataset = Dataset([
    Example([0,1,2,3,4], ["Ensolarado","Fria","Alta","Verdadeiro","Nao"]),
    Example([0,1,2,3,4], ["Nublado","Fria","Alta","Falso","Sim"]),
    Example([0,1,2,3,4], ["Ensolarado","Quente","Alta","Verdadeiro","Talvez"]),
    Example([0,1,2,3,4], ["Chuvoso","Amena","Normal","Falso","Nao"]),
    Example([0,1,2,3,4], ["Chuvoso","Quente","Normal","Verdadeiro","Sim"]),
    Example([0,1,2,3,4], ["Nublado","Amena","Alta","Verdadeiro","Talvez"]),
    Example([0,1,2,3,4], ["Nublado","Fria","Normal","Falso","Sim"]),
    Example([0,1,2,3,4], ["Chuvoso","Amena","Alta","Verdadeiro","Sim"]),
    Example([0,1,2,3,4], ["Ensolarado","Amena","Normal","Verdadeiro","Talvez"])
])

performance = ModelPerformance(node, test_dataset)

matrix = performance.confusion_matrix

table = ConfusionMatrixToConfusionTable(matrix).confusion_table_for("Nao")
print(table.true_positives, table.false_positives)
print(table.false_negatives, table.true_negatives)
print(table.precision(), table.recall(), table.f_score())

table = ConfusionMatrixToConfusionTable(matrix).confusion_table_for("Sim")
print(table.true_positives, table.false_positives)
print(table.false_negatives, table.true_negatives)
print(table.precision(), table.recall(), table.f_score())

table = ConfusionMatrixToConfusionTable(matrix).confusion_table_for("Talvez")
print(table.true_positives, table.false_positives)
print(table.false_negatives, table.true_negatives)
print(table.precision(), table.recall(), table.f_score())

print(performance.f1_measure() / 2)

predictions = list(map(lambda ex: node.predict(ex), test_dataset.examples))
true = list(map(lambda ex: ex.get_class(), test_dataset.examples))

print(f1_score(true, predictions, average='macro'))

print_tree(node)
