
from system.interfaces import FilterInterface

class HighPassFilter(FilterInterface):
    '''
    HighPassFilter allows trades where the filter values are above the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Filters assume that the supplied values are a DataElement object.
        threshold specifies the value above which the filter passes.
        '''
        self.values = values
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.values.loc[trade.entry, trade.ticker]
        if value is None:
            return False
        else:
            return value > self.threshold

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)


class LowPassFilter(FilterInterface):
    '''
    LowPassFilter allows trades where the filter values are below the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Filters assume that the supplied values are a DataElement object.
        threshold specifies the value below which the filter passes.
        '''
        self.values = values
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.values.loc[trade.entry, trade.ticker]
        if value is None:
            return False
        else:
            return value < self.threshold

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)


class BandPassFilter(FilterInterface):
    '''
    BandPassFilter allows trades according to where the filter values lie with respect to the
    specified range. if 'within_bounds' is True (default), values falling within the range result in
    acceptance of the trade, and vice-versa.
    '''
    def __init__(self, values, bounds, within_bounds = True):
        '''
        Assumes that values is a DataElement object.'
        bounds is an iterable (e.g. tuple) which specifies the upper and lower range.
        When within_bounds is True (the default), the filter will pass values between
        the min and max values of bounds, else it will pass when outside the min or max.
        '''
        self.values = values
        self.bounds = bounds
        self.within_bounds = within_bounds
        self.filter_name = values.name
        self.name = '{}:{}-{}'.format(values.name, min(bounds), max(bounds))

    def accepted_trade(self, trade):
        '''
        Returns True if the value is within the threshold at the trade entry, else False.
        '''
        value = self.values.loc[trade.entry, trade.ticker]
        if value is None:
            return False
        else:
            return (not self.within_bounds) ^ (min(self.bounds) < value < max(self.bounds))

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)


# TODO EntryLagFilter is an Entry Condition, not a Filter

class EntryLagFilter(FilterInterface):

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, strategy):
        prices = strategy.get_trade_prices()
        new_trades = []
        for trade in strategy.trades.as_list():
            if trade.duration <= self.lag:
                continue
            new_possible_entries = trade.normalised.index[trade.normalised > 0]
            new_possible_entries = new_possible_entries[new_possible_entries > self.lag]
            if len(new_possible_entries) == 0:
                continue
            new_entry_lag = min(new_possible_entries) + 1
            if new_entry_lag >= trade.duration:
                continue
            new_entry = prices[trade.entry:].index[new_entry_lag]
            new_trades.append(Trade(trade.ticker, new_entry, trade.exit, prices[trade.ticker]))
        return TradeCollection(new_trades)

    def plot(self, ticker, start, end, ax):
        pass
