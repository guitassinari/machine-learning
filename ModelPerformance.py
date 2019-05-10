class ModelPerformance:
    def __init__(self, model, test_set):
        # Model pode ser forest ou decision_tree
        # model deve implementar o método predict

        self.model = model
        self.test_set = test_set
        pass

    def f1_measure(self):
        return 5