
from . import SignalElement

class RelativeReturns(SignalElement):
    
    def __init__(self, period):
        self.period = period

    def __call__(self, prices):
        returns = (prices / prices.shift(1)) - 1
        relative = returns.subtract(returns.mean(axis = 'columns'), axis = 'rows')
        return relative.ewm(span = self.period).mean()
