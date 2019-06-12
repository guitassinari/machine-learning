import csv


class InitialWeights:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        weights = []
        weights_bias = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                # print("vetores", row)
                vectors = csv.reader(row, delimiter=',')
                for index in vectors:
                    weights.append(index)
                    # print("weights", weights)
        # print("WEIGHTS", weights)
        cont = 0
        for ite in weights:
            weights_bias.append(weights[cont][0])
            del(weights[cont][0])
            cont += 1
            # print("cont", cont)
            # print("weights bias", weights_bias)
            # print("weights new", weights)
        # print("WEIGHTS FINAL", weights)
        # print("WEIGHTS BIAS", weights_bias)
        return weights, weights_bias
