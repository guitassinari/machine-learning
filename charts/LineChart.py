import matplotlib.pyplot as plt
from itertools import count
import pandas as pd

class LineChart:
    _ids = count(0)

    def __init__(self, series, labels, max=None, min=None):
        self.id = next(self._ids)
        self.figure = plt.figure(self.id)
        if max is not None and min is not None:
            plt.ylim((min, max))
        plt.xticks(range(len(labels)), labels)
        self.series = series

        for serie in self.series:
            ts = pd.Series(serie)
            ts.plot()

    @classmethod
    def show_charts(cls):
        plt.show()




