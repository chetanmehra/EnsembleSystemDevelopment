
"""
These forecasters are inspired by the book Systematic Trading Development by Robert Carver.
They produce a normalised forecast based on the provided measurements. Forecasts range from 
-20/20 with a target absolute mean of 10.
"""
from pandas import Panel

from system.interfaces import SignalElement
from data_types.signals import Signal
from measures.moving_averages import EMA


class EwmacFamily(SignalElement):

    def __init__(self, vol_method, par_pairs):
        self.vol = vol_method
        self.par_pairs = par_pairs

    def execute(self, strategy):
        measures = {}
        for pars in self.par_pairs:
            name = "ewmac_{}_{}".format(max(pars), min(pars))
            ewmac = EWMAC(EMA(max(pars)), EMA(min(pars)), self.vol)
            measures[name] = ewmac(strategy).data
        measures = Panel(measures)
        return Signal(measures.mean(axis = 'items'), [-20, 20], measures)


class EWMAC(SignalElement):
    """
    Exponentially Weighted Moving Average Crossover.
    Creates a forecast based on the separation between two EMA measures.
    The measures are normalised based on the volatility.
    """

    def __init__(self, slow, fast, vol_method):
        self.slow = slow
        self.fast = fast
        self.vol = vol_method

    def get_result(self, strategy):
        return None

    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        slow = self.slow(prices)
        fast = self.fast(prices)
        vol = self.vol(prices)
        forecast = self.normalise((fast - slow) / vol)
        return Signal(forecast, [-20, 20], Panel({"Fast" : fast, "Slow" : slow, "Volatility" : vol}))


    def normalise(self, forecast):
        """
        Normalises the forecast to be between -20/20 with absolute average of 10.
        """
        overall_mean = forecast.abs().mean().mean()
        adjustment = forecast.abs()
        adjustment.iloc[0, :] = overall_mean
        adjustment = adjustment.ewm(200).mean()
        forecast = forecast * (10 / adjustment)
        forecast[forecast > 20] = 20
        forecast[forecast < -20] = -20
        return forecast






