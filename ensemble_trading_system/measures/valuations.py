"""
These measures all deal with fundamental valuations of the underlying business
as well as potentially incorporating the current price level(s)
"""

class ValueRatio:
    """
    ValueRatio provides the relative value at the current price. A ValueRatio
    of zero indicates fair value, negative values is overvalued, and positive are
    overvalued.
    """
    def __init__(self, values):
        self.values = values
        self.name = values.name + "_ratio"

    def __call__(self, prices):
        ratios = self.values.data / prices.data - 1
        ratios.name = self.name
        return ratios


class ValueRank(ValueRatio):
    """
    ValueRank returns the relative rank at a given day between each security, 
    based on the valuations provided and the price at the given day.
    """

    def __init__(self, values):
        self.values = values
        self.name = values.name + "_rank"

    def __call__(self, prices):
        ratios = super().__call__(prices)
        ranks = ratios.rank(axis = 1, ascending = False)
        ranks.name = self.name
        return ranks
