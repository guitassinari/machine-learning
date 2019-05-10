class Dataset:
    def __init__(self, examples):
        self.examples = examples

    def get_classes(self):
        return list(map(lambda example: example.get_class(), self.examples))

    def get_bodies(self):
        return list(map(lambda example: example.get_body(), self.examples))

    def get_attr_value(self, attr_name):
        return list(map(lambda example: example.get_attr_value(attr_name),
                        self.examples))

