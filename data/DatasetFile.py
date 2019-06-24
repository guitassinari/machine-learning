import csv
from data.Dataset import Dataset
from data.Example import Example


class DatasetFile:
    """
    Classe responsável por ler um arquivo csv contendo os dados de um dataset

    A constante CLASS_AT define o indeice do atributo onde se encontra a classe
    das instâncias, e deve ser alterado sempre que o dataset for trocado.
    """

    def __init__(self, file_path,  class_position, normalization, ignore_columns=[]):
        self.file_path = file_path
        self.class_position = class_position
        self.ignore_columns = ignore_columns
        self.normalization = normalization

    def read(self):
        examples = []
        indexes_to_remove = [self.class_position] + self.ignore_columns
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                    klass = row[self.class_position]
                    attributes = row.copy()
                    for index in sorted(indexes_to_remove, reverse=True):
                        del attributes[index]
                    attributes_and_klass = attributes + [klass]
                    examples.append(Example(list(range(len(attributes_and_klass))), attributes_and_klass))
        dataset = Dataset(examples)
        if self.normalization:
            dataset.data_normalization()

        return dataset
