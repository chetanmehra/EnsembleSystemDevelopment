
from pandas import Panel, DataFrame

from system.interfaces import SignalElement
from data_types.signals import Signal


class Crossover(SignalElement):
    '''
    The Crossover signal compares two indicators (fast and slow) to determine whether
    the prevailing trend is 'Up' (fast > slow) or 'Down' (fast <= slow).
    '''
    def __init__(self, slow, fast):
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return "x".join([self.fast.name, self.slow.name])

    @property
    def measures(self):
        return [self.fast, self.slow]

    def execute(self, strategy):
        prices = strategy.indicator_prices
        fast_ema = self.fast(prices)
        slow_ema = self.slow(prices)

        ind_data = strategy.get_empty_dataframe(fill_data = 'Down')
        ind_data[fast_ema > slow_ema] = 'Up'

        return Signal(ind_data, ['Up', 'Down'], {'Fast':fast_ema, 'Slow':slow_ema})
    
    def update_param(self, new_params):
        self.slow.update_param(max(new_params))
        self.fast.update_param(min(new_params))


class TripleCrossover(SignalElement):
    '''
    The Triple Crossover signal is similar to the Crossover except it uses three indicators (fast, mid and slow).
    The prevailing trend is 'Up' when (fast > mid) and (mid > slow), and 'Down' otherwise.
    '''
    def __init__(self, slow, mid, fast):
        self.fast = fast
        self.mid = mid
        self.slow = slow

    @property
    def name(self):
        return "x".join([self.fast.name, self.mid.name, self.slow.name])

    @property
    def measures(self):
        return [self.fast, self.mid, self.slow]
        
    def execute(self, strategy):
        prices = strategy.indicator_prices
        fast_ema = self.fast(prices)
        mid_ema = self.mid(prices)
        slow_ema = self.slow(prices)
        levels = (fast_ema > mid_ema) & (mid_ema > slow_ema)

        ind_data = strategy.get_empty_dataframe(fill_data = 'Down')
        ind_data[levels] = 'Up'
        return Signal(ind_data, ['Up', 'Down'], {'Fast':fast_ema, 'Mid':mid_ema, 'Slow':slow_ema})
    
    def update_param(self, new_params):
        pars = list(new_params)
        pars.sort()
        self.fast = pars[0]
        self.mid = pars[1]
        self.slow = pars[2]


class Breakout(SignalElement):

    def __init__(self, breakout_measure):
        self.breakout = breakout_measure

    @property
    def name(self):
        return "Breakout." + self.breakout.name

    @property
    def measures(self):
        return [self.breakout]

    def execute(self, strategy):
        prices = strategy.indicator_prices
        breakout = self.breakout(prices)
        high = breakout["high"]
        low = breakout["low"]

        ind_data = strategy.get_empty_dataframe()
        # Note: dataframe.where returns the existing values where the
        #       mask is True, and replaces them with other where False.
        #       So we need to invert the mask.
        #       Counterintuitive I think...
        ind_data = ind_data.where(~(prices.data == high), 'Up')
        ind_data = ind_data.where(~(prices.data == low), 'Down')
        ind_data = ind_data.ffill()
        # We need to remove any remaining Nans (at the start of the df), 
        # otherwise later string comparisons will throw an error because
        # it thinks it is a mixed datatype.
        ind_data = ind_data.fillna('-')

        return Signal(ind_data, ['Up', 'Down'], breakout)
    
    def update_param(self, new_params):
        self.breakout.update_param(new_params)



class MultiLevelMACD(SignalElement):
    """
    The MultiLevelMACD creates a defined set of levels above and below zero
    based on the MACD level.
    """
    def __init__(self, fast, slow, levels):
        """
        ema_pars is a tuple of (fast, slow) parameters for the EMA calc.
        levels is an integer which determines how many segments the 
        signal is broken up into above and below zero.
        """
        self.fast = fast
        self.slow = slow
        self.levels = levels

    @property
    def name(self):
        return "{}.lvl-MACD_".format(self.levels) + ".".join(self.fast.name, self.slow.name)

    def update_params(self, new_params):
        self.fast.update_param(min(new_params))
        self.slow.update_param(max(new_params))

    def execute(self, strategy):
        prices = strategy.indicator_prices
        fast = self.fast(prices)
        slow = self.slow(prices)
        return Signal(fast - slow, ['forecast'], {'fast':fast, 'slow':slow})


