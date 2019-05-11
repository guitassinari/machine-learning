import csv
from data.Dataset import Dataset
from data.Example import Example


class DatasetFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        attr_names = None
        examples = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            first_row = True
            for row in csv_reader:
                if first_row:
                    attr_names = row
                    first_row = False
                else:
                    examples.append(Example(attr_names, row))
        return Dataset(examples)


        # ler o arquivo em file_path e retornar um Dataset
