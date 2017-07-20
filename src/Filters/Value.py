
from System.Filter import FilterInterface, StackedFilterValues


class ValueFilterValues(StackedFilterValues):
    '''
    Returns the ratio of the calculated value to the price at time of entry.
    '''
    def get_for_df(self, df, row):
        ticker = df.ticker[row]
        date = df.entry[row]
        recent = self.get(ticker, date)
        price = df.entry_price[row]
        if recent is not None:
            return recent / price
        else:
            return recent


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
