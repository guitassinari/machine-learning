import csv
from data.Dataset import Dataset
from data.Example import Example


class DatasetFile:
    CLASS_AT = -1

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
