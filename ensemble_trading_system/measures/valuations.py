"""
These measures all deal with fundamental valuations of the underlying business
as well as potentially incorporating the current price level(s)
"""

from . import SignalElement

class ValueRatio(SignalElement):
    """
    ValueRatio provides the relative value at the current price. A ValueRatio
    of zero indicates fair value, negative values is overvalued, and positive are
    overvalued.
    """
    def __init__(self, valuation_type):
        self.valuation_type = valuation_type

    @property
    def name(self):
        return self.valuation_type + "_ratio"

    def __call__(self, strategy):
        values = strategy.market.get_valuations(self.valuation_type)
        prices = strategy.indicator_prices
        ratios = values.data / prices.data - 1
        ratios.name = self.name
        return ratios


class ValueRank(ValueRatio):
    """
    ValueRank returns the relative rank at a given day between each security, 
    based on the valuations provided and the price at the given day.
    """

    def __init__(self, values):
        self.values = values

    @property
    def name(self):
        return self.valuation_type + "_rank"

    def __call__(self, strategy):
        ratios = super().__call__(strategy)
        ranks = ratios.rank(axis = 1, ascending = False)
        ranks.name = self.name
        return ranks
