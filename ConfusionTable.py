class ConfusionTable:
    BETA = 1

    def __init__(self, tp, tn, fp, fn):
        self.true_positives = tp
        self.true_negatives = tn
        self.false_positives = fp
        self.false_negatives = fn
        print(tp, tn, fp, fn)

    def f_score(self):
        if self.precision() == 0: # evitar divisão por 0
            return 0
        return (
            (1+self.BETA**2)*self.precision()
        )/(
            (self.BETA**2)*self.precision() + self.recall()
        )

    def precision(self):
        if self.true_positives == 0: # evitar divisão por 0
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        if self.true_positives == 0: # evitar divisão por 0
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)

