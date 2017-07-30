
"""
These forecasters are inspired by the book Systematic Trading Development by Robert Carver.
They produce a normalised forecast based on the provided measurements. Forecasts range from 
-20/20 with a target absolute mean of 10.
"""

from system.interfaces import StrategyElement


class EWMAC(StrategyElement):
    """
    Exponentially Weighted Moving Average Crossover.
    Creates a forecast based on the separation between two EMA measures.
    The measures are normalised based on the volatility.
    """

    def __init__(self, slow, fast, vol_method):
        self.slow = slow
        self.fast = fast
        self.vol = vol_method


    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        slow = self.slow(prices)
        fast = self.fast(prices)
        vol = self.vol(prices)
        forecast = (fast - slow) / vol

        return forecast




