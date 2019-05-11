import matplotlib.pyplot as plt
from itertools import count
import pandas as pd


class LineChart:
    _ids = count(0)

    def __init__(self, series):
        self.id = next(self._ids)
        self.figure = plt.figure(self.id)
        self.series = series

        for serie in self.series:
            ts = pd.Series(serie)
            ts.plot()

    @classmethod
    def show_charts(cls):
        plt.show()




