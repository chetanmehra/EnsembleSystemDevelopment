
from System.Filter import FilterInterface, StackedFilterValues
from System.Trade import Trade, TradeCollection



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


    def __call__(self, strategy):
        return strategy.trades.find(self.accepted_trade)


    def plot(self, ticker, start, end, ax):
        values = self.values.plot(ticker, start, end, ax)


class ValueRankFilter(FilterInterface):

    def __init__(self, values, market, max_rank):
        self.ranks = values.value_rank(market)
        self.max_rank = max_rank

    def accepted_trade(self, trade):
        rank = self.ranks.loc[trade.entry, trade.ticker]
        return rank <= self.max_rank

    def __call__(self, strategy):
        return strategy.trades.find(self.accepted_trade)





