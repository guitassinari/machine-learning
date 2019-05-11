import json


class HyperParametersFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            return data["parameters"]
