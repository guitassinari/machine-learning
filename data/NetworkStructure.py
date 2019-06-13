import csv


class NetworkStructure:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        examples = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\n')
            for row in csv_reader:
                examples.append(row)
        return examples
