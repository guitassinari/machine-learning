class Example:
    def __init__(self, attr_names, attr_values):
        self.attr_names = attr_names
        self.attr_values = attr_values

    def get_attr_value(self, attr_name):
        attr_index = self.attr_names.index(attr_name)
        return self.attr_values[attr_index]

    def get_class(self):
        return self.attr_values[-1]

    def get_body(self):
        return self.attr_values[0:-1]
