from sklearn.utils import resample


class Dataset:
    """Classe que engloba um Dataset (lista de instâncias / Examples)

    A classe Dataset foi criada apenas para se tornar um wrapper no entorno
    de um dataset comum.
    Encaramos um dataset como uma lista de instâncias. Cada instância é encapsulada
    pela classe Example. Para melhores informações, ver a documentação em data/Example.
    """
    def __init__(self, examples, attr_names=None, possible_classes=None, attr_values=None):
        self.__major_class = None
        self.__all_classes = None
        self.__all_bodies = None
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
        if self.__all_classes is None:
            self.__all_classes = list(map(lambda example: example.get_class(), self.examples))
        return self.__all_classes

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
        if self.__all_bodies is None:
            self.__all_bodies = list(map(lambda example: example.get_body(), self.examples))

        return self.__all_bodies

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
        """
        Retorna todos os valores possíveis para o atributo attr_name
        :param attr_name: nome/identificador do atributo que se deseja os valores possíveis
        :return: uma lista contendo todos os possíveis valores para o atributo requisitado
        """
        attr_index = self.attr_names.index(attr_name)
        return self.attr_values[attr_index].copy()

    def __get_uniq_attr_values_from_examples(self, attr_name):
        """
        Calcula, a partir dos exemplos do dataset, todos os valores possíveis do
        atributo attr_name. Utilizado na inicialização do dataset
        :param attr_name: nome do atributo para o qual se deseja os valores
        :return: todos os valores possíveis do atributo
        """
        unique_list = []
        all_values = self.get_attr_value(attr_name)
        for value in all_values:
            if value not in unique_list:
                unique_list.append(value)
        return unique_list

    def __uniq_classes(self):
        """
        A partir das classes dos exemplos, calcula todos os possiveis valores
        para as classes
        :return: lista com todos os valores possíveis de classes
        """
        unique_list = []
        all_classes = self.get_classes()
        for klass in all_classes:
            if klass not in unique_list:
                unique_list.append(klass)
        return unique_list

    def get_attr_names(self):
        """
        Nome/identificadores dos atributos
        :return: nome dos atributos
        """
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
        """
        Retorna o tipo de um determinado atributo (attr_name), podendo ser
        AttributeType.CATEGORIC ou AttributeType.NUMERIC. Ver Example e
        AttributeType

        :param attr_name: nome/identificador do atributo
        :return: Tipo de atributo: AttributeType.CATEGORIC  ou AttributeType.NUMERIC
        """
        return self.examples[0].get_attr_type(attr_name)

    def major_class(self):
        """
        retorna a classe de maior frequência no dataset e a memoriza para
        chamadas posteriores

        :return: classe mais frequente
        """
        if self.__major_class is None:
            self.__major_class = self.__find_major_klass()
        return self.__major_class

    def __find_major_klass(self):
        """
        Calcula qual é a classe mais frequente no dataset

        :return: classe mais frequente no dataset
        """
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
        """
        Retorna um subset com apenas os exemplos onde o atributo attr_name possui
        o valor attr_value

        :param attr_name: identificador do atributo
        :param attr_value: valor do atributo
        :return: retorna o dataset onde o atributo possui o valor requisitado
        """
        new_dataset = []
        for i in range(len(self.examples)):
            example = self.examples[i]
            if example.get_attr_value(attr_name) == attr_value:
                new_dataset.append(i)
        return self.subset(new_dataset)

    def empty(self):
        """
        Retorna se o dataset está vazio (não possui exemplos), ou não.
        :return: true se o dataset não tiver possuir exemplos. false caso o contrário
        """
        return len(self.examples) == 0

    def pure(self):
        """
        Calcula se o dataset possui apenas uma classe presente em seus exemplos
        :return: True se os exemplos do dataset tiverem apenas um classe. False caso o contrário
        """
        classes = self.get_classes()
        first_klass = classes[0]
        return classes.count(first_klass) == len(classes)

    def resample(self, sample_size):
        """
        realiza uma amostragem estratificada com reposição de tamnho sample_size
        :param sample_size: tamanho da amostragem
        :return: um dataset contendo os exemplos amostrados
        """
        examples = resample(self.examples,
                                n_samples=sample_size,
                                stratify=self.get_classes())
        return Dataset(examples,
                       self.attr_names.copy(),
                       self.possible_classes.copy(),
                       self.attr_values.copy())

    def subset(self, examples_indexes):
        """
        Cria um subset com os exemplos representados pelos indices recebidos
        :param examples_indexes: indices dos exemplos que se deseja no subset
        :return: o subset com os exemplos requisitados
        """
        examples = self.__get_examples_by_multiple_indexes(examples_indexes)
        return Dataset(examples,
                       self.attr_names.copy(),
                       self.possible_classes.copy(),
                       self.attr_values.copy())

    def __get_examples_by_multiple_indexes(self, set_indexes):
        """
        Função auxiliar para retornar uma lista de exemplos a partir de seus indices
        :param set_indexes: indices dos exemplos que se deseja no subset
        :return: a lista com os exemplos requisitados
        """
        # transforma em array numpy pra poder pegar exemplos com uma lista de indices
        filtered_examples = []
        for index in set_indexes:
            filtered_examples.append(self.get_example_at(index))
        return filtered_examples

    def get_example_at(self, index):
        """
        retorna um exemplo de acordo com seu indice
        :param index: indice do exemplo desejado
        :return: o exemplo desejado
        """
        return self.examples[index]
