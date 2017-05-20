
from System.Strategy import MeasureElement
from System.Indicator import LevelIndicator


class TrueFalseCross(MeasureElement):

    def __init__(self, trend_signal):
        self.trend_signal = trend_signal

    @property
    def name(self):
        return ".".join(["TrueFalseX", self.trend_signal.name])

    def execute(self, strategy):
        indicator = self.trend_signal.execute(strategy)
        return LevelIndicator(indicator.data.astype(str), indicator.measures)


