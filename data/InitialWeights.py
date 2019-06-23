import csv


class InitialWeights:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path) as csv_file:
            bias_matrices = []
            weight_matrices = []
            csv_reader = csv.reader(csv_file, delimiter='\n')
            for layer_weights_strings in csv_reader:
                line_string = layer_weights_strings[0]
                columns = line_string.split(';')
                weight_matrix = []
                bias_matrix = []
                for weights_string in columns:
                    weights = list(map(lambda weight_string: float(weight_string), weights_string.split(',')))
                    bias = weights.pop(0)
                    bias_matrix.append([bias])
                    weight_matrix.append(weights)
                bias_matrices.append(bias_matrix)
                weight_matrices.append(weight_matrix)
        return weight_matrices, bias_matrices
