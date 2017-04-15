
from System.Position import PositionSelectionElement, FilteredPositions

class Filter(PositionSelectionElement):

    def __init__(self, values, filter_range):
        '''
        Requires that values is a DataFrame with first column 'ticker' and next column filter values.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        self.values = values
        self.bounds = filter_range
        self.left = min(filter_range)
        self.right = max(filter_range)
        self.name = values.name


    def execute(self, strategy):
        '''
        Removes position values where filter criteria is not met.
        '''
        trades = strategy.positions.trades.as_list()
        trade_frame = strategy.positions.trades.as_dataframe_with(self.values)
        trade_frame["Filter"] = trade_frame[self.name] / trade_frame.entry_price
        filtered_bools = (trade_frame.Filter > self.left) & (trade_frame.Filter <= self.right)

        accepted_trades = []
        eliminated_trades = []

        for i, keep in enumerate(filtered_bools.values):
            if keep:
                accepted_trades.append(trades[i])
            else:
                eliminated_trades.append(trades[i])

        new_positions = strategy.positions.copy()
        new_positions.remove(eliminated_trades)

        return FilteredPositions(strategy.positions, new_positions.data, accepted_trades, new_positions.tickers)


class FilterValues:

    def __init__(self, values, name = None):
        '''
        values should be a dataframe with index of dates
        '''
        self.values = values
        if name is None:
            self.name = "Filter"
        else:
            self.name = name


    def __getitem__(self, key):
        raise NotImplementedError

    def get(self, ticker, date):
        tick_values = self[ticker]
        relevant_values = tick_values[tick_values.index < date]
        if relevant_values.empty:
            result = None
        elif relevant_values.shape[1] == 1:
            result = relevant_values.iloc[-1].values[0]
        else:
            result = relevant_values.iloc[-1].values
        return result


class StackedFilterValues(FilterValues):

    def __getitem__(self, key):
        return self.values[self.values.ticker == key][self.values.columns[1:]]

    @property
    def types(self):
        return self.values.columns[1:].tolist()


class WideFilterValues(FilterValues):

    def __getitem__(self, key):
        return self.values[[key]]

    @property
    def types(self):
        return [self.name]

