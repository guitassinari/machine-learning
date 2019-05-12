from sklearn.utils import resample
from models.InfoGain import InfoGain


class Dataset:
    """Classe que engloba um Dataset (lista de instâncias / Examples)

    A classe Dataset foi criada apenas para se tornar um wrapper no entorno
    de um dataset comum.
    Encaramos um dataset como uma lista de instâncias. Cada instância é encapsulada
    pela classe Example. Para melhores informações, ver a documentação em data/Example.
    """
    def __init__(self, examples, attr_names=None, possible_classes=None, attr_values=None):
        self.__major_class = None
        self.examples = examples
        self.attr_names = attr_names or examples[0].get_attr_names()
        self.possible_classes = possible_classes or self.__uniq_classes()
        self.attr_values = attr_values or\
                           list(map(lambda attr_name: self.__get_uniq_attr_values_from_examples(attr_name), self.attr_names))

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
        return self.possible_classes.copy()

    def get_uniq_attr_values(self, attr_name):
        attr_index = self.attr_names.index(attr_name)
        return self.attr_values[attr_index].copy()

    def __get_uniq_attr_values_from_examples(self, attr_name):
        unique_list = []
        all_values = self.get_attr_value(attr_name)
        for value in all_values:
            if value not in unique_list:
                unique_list.append(value)
        return unique_list

    def __uniq_classes(self):
        unique_list = []
        all_classes = self.get_classes()
        for klass in all_classes:
            if klass not in unique_list:
                unique_list.append(klass)
        return unique_list

    def get_attr_names(self):
        return self.attr_names.copy()

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

    def get_attr_type(self, attr_name):
        return self.examples[0].get_attr_type(attr_name)

    def major_class(self):
        if self.__major_class is None:
            self.__major_class = self.__find_major_klass()
        return self.__major_class

    def __find_major_klass(self):
        classes = self.get_classes()
        max_frequency_so_far = 0
        major = classes[0]
        for klass in classes:
            klass_frequency = classes.count(klass)
            if klass_frequency > max_frequency_so_far:
                max_frequency_so_far = klass_frequency
                major = klass
        return major

    def split_dataset_for(self, attr_name, attr_value):
        new_dataset = []
        for i in range(len(self.examples)):
            example = self.examples[i]
            if example.get_attr_value(attr_name) == attr_value:
                new_dataset.append(i)
        return self.subset(new_dataset)

    def empty(self):
        return len(self.examples) == 0

    def pure(self):
        classes = self.get_classes()
        first_klass = classes[0]
        return classes.count(first_klass) == len(classes)

    def resample(self, sample_size):
        examples = resample(self.examples,
                                n_samples=sample_size,
                                stratify=self.get_classes())
        return Dataset(examples,
                       self.attr_names.copy(),
                       self.possible_classes.copy(),
                       self.attr_values.copy())

    def subset(self, examples_indexes):
        examples = self.__get_examples_by_multiple_indexes(examples_indexes)
        return Dataset(examples,
                       self.attr_names.copy(),
                       self.possible_classes.copy(),
                       self.attr_values.copy())

    def __get_examples_by_multiple_indexes(self, set_indexes):
        # transforma em array numpy pra poder pegar exemplos com uma lista de indices
        filtered_examples = []
        for index in set_indexes:
            filtered_examples.append(self.get_example_at(index))
        return filtered_examples

    def get_example_at(self, index):
        return self.examples[index]
