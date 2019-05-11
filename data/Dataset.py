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

    def get_uniq_classes(self):
        unique_list = []
        all_classes = self.get_classes()
        for klass in all_classes:
            if klass not in unique_list:
                unique_list.append(klass)
        return unique_list

    def get_examples(self):
        return self.examples.copy()

    def size(self):
        return len(self.examples)

