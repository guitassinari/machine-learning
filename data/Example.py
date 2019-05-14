from data.Attribute import AttributeType
"""
Uma instância é um conjunto de atributos e uma classe
{
    nome: "Joao"
    idade: 2,
    joga: "Nao"  # Esta é a classe
}
"""


class Example:
    """Classe que encapsula uma instância
    """
    def __init__(self, attr_names, attr_values):
        """
        :param attr_names: lista com nome de todos os atributos. O último é o nome da classe
        :param attr_values: lista com todos os valores dos atributos. O último é o valor da classe
        """
        self.attr_names = attr_names
        self.attr_values = attr_values

    def get_attr_value(self, attr_name):
        """
        :param attr_name: nome do atributo que se deseja pegar o valor
        :return: valor do atributo na instância
        """
        attr_index = self.attr_names.index(attr_name)
        return self.attr_values[attr_index]

    def get_class(self):
        """
        :return: o valor da classe da inst￿ância
        """
        return self.attr_values[-1]

    def get_body(self):
        """
        :return: valores de todos os atributos, menos o da classe
        """
        return self.attr_values[0:-1]

    def get_attr_names(self):
        """
        Retorna o nome de todos os atributos do exemplo
        :return: nome de todos os atributos do exemplo
        """
        return self.attr_names[0:-1].copy()

    def get_attr_type(self, attr_name):
        """
        Retorna o tipo de um atributo, podendo ser AttributeType.CATEGORIC ou
        AttributeType.NUMERIC
        :param attr_name: nome do atributo cujo tipo se deseja conseguir
        :return: tipo do atributo
        """
        value = self.get_attr_value(attr_name)
        try:
            float(value)
            return AttributeType.NUMERIC
        except ValueError:
            return AttributeType.CATEGORIC
