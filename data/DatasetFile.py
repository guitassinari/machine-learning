import csv
from data.Dataset import Dataset
from data.Example import Example


class DatasetFile:
    """
    Classe responsável por ler um arquivo csv contendo os dados de um dataset

    A constante CLASS_AT define o indeice do atributo onde se encontra a classe
    das instâncias, e deve ser alterado sempre que o dataset for trocado.
    """
    CLASS_AT = 0

    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        examples = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                    klass = row[self.CLASS_AT]
                    attributes = row.copy()
                    del attributes[self.CLASS_AT]
                    examples.append(Example(list(range(len(row))), attributes + [klass]))
        return Dataset(examples)
