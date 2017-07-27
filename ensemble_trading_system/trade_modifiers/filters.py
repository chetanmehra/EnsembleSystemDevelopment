
from system.interfaces import FilterInterface



class ValueRangeFilter(FilterInterface):
    '''
    ValueRangeFilter filters strategy trades based on the valuation at the time of
    entry being within the specified range.
    '''
    def __init__(self, values, filter_range):
        '''
        Requires that values is a ValueFilterValues object.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        self.values = values
        self.left = min(filter_range)
        self.right = max(filter_range)
        self.filter_name = values.name
        self.name = '{}:{}-{}'.format(values.name, *filter_range)

    def accepted_trade(self, trade):
        '''
        Returns True if the trade meets the filter criterion, else False.
        '''
        value = self.values.get(trade.ticker, trade.entry)
        try:
            ratio = value / trade.entry_price
        except TypeError:
            return False
        else:
            return (ratio > self.left) and (ratio <= self.right)

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)


class ValueRankFilter(FilterInterface):
    '''
    ValueRankFilter filters trades based on the rank of the valuation calculated at the
    time of trade entry.
    '''
    def __init__(self, values, market, max_rank):
        self.ranks = values.value_rank(market)
        self.max_rank = max_rank

    def accepted_trade(self, trade):
        '''
        Returns True if the trade meets the filter criterion, else False.
        '''
        rank = self.ranks.loc[trade.entry, trade.ticker]
        return rank <= self.max_rank


class HighPassFilter(FilterInterface):
    '''
    HighPassFilter allows trades where the filter values are above the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Requires that values is a WideFilterValues object.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        self.values = values
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.values.get(trade.ticker, trade.entry)
        return value > self.threshold

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)

class LoPassFilter(FilterInterface):
    '''
    LoPassFilter allows trades where the filter values are below the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Requires that values is a WideFilterValues object.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        self.values = values
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.values.get(trade.ticker, trade.entry)
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
        Requires that values is a WideFilterValues object.
        filter_range should be a tuple representing the acceptable filter range.
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
        value = self.values.get(trade.ticker, trade.entry)
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
