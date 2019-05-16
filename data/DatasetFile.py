import csv
from data.Dataset import Dataset
from data.Example import Example


class DatasetFile:
    """
    Classe responsável por ler um arquivo csv contendo os dados de um dataset

    A constante CLASS_AT define o indeice do atributo onde se encontra a classe
    das instâncias, e deve ser alterado sempre que o dataset for trocado.
    """

    def __init__(self, file_path,  class_position):
        self.file_path = file_path
        self.class_position = class_position

    def read(self):
        examples = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                    klass = row[self.class_position]
                    attributes = row.copy()
                    del attributes[self.class_position]
                    examples.append(Example(list(range(len(row))), attributes + [klass]))
        return Dataset(examples)
