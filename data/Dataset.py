class Dataset:
    """Classe que engloba um Dataset (lista de instâncias / Examples)

    A classe Dataset foi criada apenas para se tornar um wrapper no entorno
    de um dataset comum.
    Encaramos um dataset como uma lista de instâncias. Cada instância é encapsulada
    pela classe Example. Para melhores informações, ver a documentação em data/Example.
    """
    def __init__(self, examples):
        self.examples = examples

    def get_classes(self):
        """
        Retorna as classes de todas as instâncias do dataset


        Type: Array of [string, number]
        Exemplo:

        ["sim", "nao", "sim", "sim", "nao"]
        """
        return list(map(lambda example: example.get_class(), self.examples))

    def get_bodies(self):
        """
        Retorna os atributos de todas as instâncias do dataset em forma de lista
        de hashes/dict

        Type: Array of [dictionary]
        Exemplo:
        [
            {nome: "junior", idade: 2},
            {nome: "sandy", idade: 5},
            {nome: "alice", idade: 99},
        ]
        """
        return list(map(lambda example: example.get_body(), self.examples))

    def get_attr_value(self, attr_name):
        """
        Retorna todos os valores de um atributo de todas as instâncias

        Type: Array of [string, number]
        Exemplo:
        get_attr_value("idade")
        [2, 5, 99]
        """
        return list(map(lambda example: example.get_attr_value(attr_name),
                        self.examples))

    def get_uniq_classes(self):
        """
        Retorna uma lista com os possíveis valores para classes do Dataset

        Type: Array of [string]
        Exemplo:
        ["sim", "nao"]
        """
        unique_list = []
        all_classes = self.get_classes()
        for klass in all_classes:
            if klass not in unique_list:
                unique_list.append(klass)
        return unique_list

    def get_examples(self):
        """
        Retorna uma lista com os todos os exemplos do dataset

        Type: Array of [Example]
        Exemplo:
        [
            Example(nome: "junior", idade: 2, joga: "sim"),
            Example(nome: "sandy", idade: 5, joga: "nao"),
        ]
        """
        return self.examples.copy()

    def size(self):
        """
        Retorna quantos exemplos existem no Dataset

        Type: number
        Exemplo: 3
        """
        return len(self.examples)

