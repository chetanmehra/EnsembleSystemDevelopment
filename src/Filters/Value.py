
from System.Filter import FilterInterface
from System.Trade import Trade, TradeCollection


class ValueRangeFilter(FilterInterface):

    def __init__(self, signal, values, filter_range):
        '''
        Requires that values is a DataFrame with first column 'ticker' and next column filter values.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        super().__init__(signal)
        self.values = values
        self.left = min(filter_range)
        self.right = max(filter_range)
        self.filter_name = values.name
        self.name = '{}:{}-{}'.format(values.name, *filter_range)


    # HACK trade filtering is clunky; should use TradeCollection.find()
    def __call__(self, strategy):
        trades = strategy.trades.as_list()
        trade_frame = strategy.trades.as_dataframe_with(self.values)
        filtered_bools = (trade_frame[self.filter_name] > self.left) & (trade_frame[self.filter_name] <= self.right)

        accepted_trades = []

        for i, keep in enumerate(filtered_bools.values):
            if keep:
                accepted_trades.append(trades[i])
        
        return TradeCollection(accepted_trades)


    def plot(self, ticker, start, end, ax):
        values = self.values.plot(ticker, start, end, ax)

