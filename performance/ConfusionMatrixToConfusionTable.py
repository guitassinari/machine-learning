from performance.ConfusionTable import ConfusionTable


class ConfusionMatrixToConfusionTable:
    BETA = 1

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def confusion_table_for(self, klass):
        return ConfusionTable(
                self.__true_positives_for(klass),
                self.__true_negatives_for(klass),
                self.__false_positives_for(klass),
                self.__false_negatives_for(klass)
            )

    def __true_positives_for(self, klass):
        predicted_hash = self.confusion_matrix.predictions_for(klass)
        return predicted_hash[klass]

    def __false_positives_for(self, klass):
        other_klasses = self.__other_klasses_than(klass)
        false_positives = 0
        predicted_hash = self.confusion_matrix.predictions_for(klass)
        for _klass in other_klasses:
            if _klass != klass:
                false_positives += predicted_hash[_klass]
        return false_positives

    def __false_negatives_for(self, klass):
        other_klasses = self.__other_klasses_than(klass)
        false_negatives = 0
        for _klass in other_klasses:
            prediction_hash = self.confusion_matrix.predictions_for(_klass)
            false_negatives += prediction_hash[klass]
        return false_negatives

    def __true_negatives_for(self, klass):
        other_klasses = self.__other_klasses_than(klass)
        true_negatives = 0
        for _klass in other_klasses:
            prediction_hash = self.confusion_matrix.predictions_for(_klass)
            true_negatives += prediction_hash[_klass]
        return true_negatives

    def __other_klasses_than(self, klass):
        different_classes = self.confusion_matrix.possible_classes()
        different_classes.remove(klass)
        return different_classes
