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
    def __init__(self, valuation_type, sub_type, price_denominator = True):
        self.valuation_type = valuation_type
        self.sub_type = sub_type
        self.price_denominator = price_denominator

    @property
    def name(self):
        return "_".join([self.valuation_type, self.sub_type, "ratio"])

    def __call__(self, strategy):
        values = strategy.market.get_valuations(self.valuation_type, self.sub_type)
        prices = strategy.indicator_prices
        if self.price_denominator:
            ratios = values.data / prices.data - 1
        else:
            ratios = prices.data / values.data
        ratios.name = self.name
        return ratios


class ValueRank(ValueRatio):
    """
    ValueRank returns the relative rank at a given day between each security, 
    based on the valuations provided and the price at the given day.
    """

    @property
    def name(self):
        return "_".join([self.valuation_type, self.sub_type, "rank"])

    def __call__(self, strategy):
        ratios = super().__call__(strategy)
        ranks = ratios.rank(axis = 1, ascending = False)
        ranks.name = self.name
        return ranks
