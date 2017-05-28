
from System.Strategy import MeasureElement
from System.Trade import Trade, TradeCollection
from System.Indicator import Indicator
from pandas import Panel


# TODO Create breakout signal generator

class Crossover(MeasureElement):
    
    def __init__(self, slow, fast):
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return "x".join([self.fast.name, self.slow.name])

    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = self.fast(prices)
        slow_ema = self.slow(prices)
        levels = fast_ema > slow_ema
        return Indicator(levels, Panel.from_dict({'Fast':fast_ema, 'Slow':slow_ema}))
    
    def update_param(self, new_params):
        self.slow.update_param(new_params[0])
        self.fast.update_param(new_params[1])


class TripleCrossover(MeasureElement):
    
    def __init__(self, slow, mid, fast):
        self.fast = fast
        self.mid = mid
        self.slow = slow

    @property
    def name(self):
        return "x".join([self.fast.name, self.mid.name, self.slow.name])
        
    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = self.fast(prices)
        mid_ema = self.mid(prices)
        slow_ema = self.slow(prices)
        levels = (fast_ema > mid_ema) & (mid_ema > slow_ema)
        return Indicator(levels, Panel.from_dict({'Fast':fast_ema, 'Mid':mid_ema, 'Slow':slow_ema}))
    
    def update_param(self, new_params):
        pars = list(new_params)
        pars.sort()
        self.fast = pars[0]
        self.mid = pars[1]
        self.slow = pars[2]


# TODO ValueWeightedEMA is not complete
class ValueWeightedEMA(MeasureElement):

    def __init__(self, values, fast, slow):
        self.values = values
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return ".".join([self.values.name, "Wtd", self.fast.name, self.slow.name])

    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        value_ratio = DataFrame(prices)
        value_ratio[:] = None
        for col in self.values:
            value_ratio[col] = self.values[col]
        value_ratio.fillna(method = 'ffill')
        value_ratio = value_ratio / prices

    def update_param(self, new_params):
        pass

