import csv


class NetworkStructure:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\n')
            layers_structure = []
            for row in csv_reader:
                value_as_string = row[0]
                float_value = float(value_as_string)
                layers_structure.append(float_value)

            _lambda = layers_structure.pop(0)

            return layers_structure, _lambda
