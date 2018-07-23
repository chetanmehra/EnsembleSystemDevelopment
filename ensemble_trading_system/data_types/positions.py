
from copy import deepcopy
from pandas import DataFrame, Series, concat
from numpy import sign, NaN

from system.interfaces import DataElement

from data_types.returns import AggregateReturns, GroupReturns
from data_types.trades import Trade, TradeCollection
from data_types.events import EventCollection


class Position(DataElement):
    '''
    Position objects hold the dual role of keeping the position data for the strategy, as well as 
    calculating the returns from holding those positions.
    '''
    def __init__(self, data, strategy = None):
        if not isinstance(data, DataFrame):
            raise TypeError
        self.data = data
        if strategy is not None:
            self.base_returns = strategy.market_returns.data
            self._returns = self.calculate_returns()
            self.events = self.create_events()
            self.trades = self.create_trades()

    @staticmethod
    def from_copy(data, trades, base_returns):
        '''
        from_copy is a constructor which is designed to be used when conducting
        trials of trade_modifiers. This will create a "lightweight" version of 
        positions, with the supplied data.
        '''
        positions = Position(data)
        positions.trades = trades
        positions.base_returns = base_returns
        return positions

    def create_events(self):
        return EventCollection.from_position_data(self.data)

    def create_trades(self):
        trades = []
        for ticker in self.events.tickers:
            # Note we loop through by ticker and subset the events here so 
            # we don't have to search the full set of tickers every time.
            # The EventCollection caches the subset for each ticker behind the
            # scenes.
            # Although less elegant, this approach reduced the calculation time 
            # significantly.
            ticker_entries = self.events.related_entries(ticker)
            ticker_exits = self.events.related_exits(ticker)
            for entry in ticker_entries:
                trade_exit = self.events.next_exit(entry, ticker_exits)
                # If an entry is generated on the last day, then the entry
                # and exit date will be the same which will cause issues.
                if entry.date < trade_exit.date:
                    trades.append(Trade(entry, trade_exit, self.data.loc, 
                                    self.base_returns.loc, self._returns.loc))
        return TradeCollection(trades)

    def calculate_returns(self):
        # TODO make sure this can handle short positions.
        return (self.base_returns * self.data.shift(1)).fillna(0)

    @property
    def returns(self):
        return GroupReturns(self._returns)

    def update_returns(self):
        '''
        update_returns recalculates the returns as per calculate_returns
        however it also updates references to the returns where needed.
        '''
        self._returns = self.calculate_returns()
        for trade in self.trades:
            trade.weighted_returns = self._returns.loc

    def copy(self):
        new_data = self.data.copy()
        new_trades = self.trades.copy()
        for trade in new_trades:
            trade._position_loc = new_data.loc
        return Position.from_copy(new_data, new_trades, self.base_returns)

    def apply(self, trade_modifier):
        '''
        apply accepts a trade_modifier, which is used to update the positions, 
        trades, and returns.
        '''
        self.trades = self.trades.apply(trade_modifier)
        self.update_returns()
        
    def trial(self, trade_modifier):
        '''
        trial accepts a trade_modifier and performs the update as per apply, 
        but returns a copy.
        '''
        new_positions = self.copy()
        new_positions.apply(trade_modifier)
        return new_positions
        
    def update_from_trades(self, trades):
        new_pos_data = deepcopy(self.data)
        new_pos_data[:] = 0
        for trade in trades.as_list():
            new_pos_data.loc[trade.entry:trade.exit, trade.ticker] = self.data.loc[trade.entry:trade.exit, trade.ticker]
        self.data = new_pos_data
        self.events = self.create_events()
        self.trades = trades

    def subset(self, subset_tickers):
        new_data = self.data[subset_tickers]
        new_positions = Position(new_data)
        new_positions.trades = self.trades.find(lambda T: T.ticker in subset_tickers)
        return new_positions

    def applied_to(self, market_returns):
        data = self.data * market_returns.data
        #data = data.div(self.num_concurrent(), axis = 'rows')
        return AggregateReturns(data)

    @property
    def start(self):
        num = self.num_concurrent()
        return num[num != 0].index[0]

    def long_only(self):
        data = deepcopy(self.data)
        data[data < 0] = 0
        return Position(data)

    def short_only(self):
        data = deepcopy(self.data)
        data[data > 0] = 0
        return Position(data)

    def remove(self, excluded):
        for trade in excluded:
            self.data.loc[trade.entry:trade.exit, trade.ticker] = 0

    def num_concurrent(self):
        '''
        Returns a series of the number of concurrent positions held over time.
        '''
        data = sign(self.data)
        return data.sum(axis = 1)

    def normalised(self):
        """
        Returns a new Position object with sizes normalised by dividing by the current
        total number of positions held on that day.
        """
        data = self.data.copy()
        data = data.div(self.num_concurrent(), axis = 0)
        return Position(data)

    def unitised(self):
        """
        Returns a new positions object with sizes set to unity (1 long, -1 long).
        """
        position_sign = (self.data > 0) * 1
        position_sign[self.data < 0] = -1
        return Position(position_sign)
        

    # TODO discretise only works for long positions at the moment
    def discretise(self, min_size, max_size, step):
        """
        Returns a new Position object of discrete position signals.
        discretise takes the continuous positions and turns it into a stepped series,
        attempting to keep the average area under the curve approximately equivalent.
        Refer: https://qoppac.blogspot.com.au/2016/03/diversification-and-small-account-size.html
        """
        i = 0
        pos = self.data.copy()
        pos[pos < min_size] = 0
        while min_size + i * step < max_size:
            lower = min_size + i * step
            upper = min_size + (i + 1) * step
            size = min_size + (i + 0.5) * step
            pos[(pos >= lower) & (pos < upper)] = size
            i += 1
        pos[pos > max_size] = max_size
        return Position(pos)
   
    def summary(self):
        trades = self.trades.summary()
        overall_returns = self.returns.summary()
        drawdowns = self.returns.summary_drawdowns()
        return concat((trades, overall_returns, drawdowns))

class Positions2:
    '''
    Derivative of Positions. Attempts to keep all data as DataFrame's
    using groupby to produce by-trade statistics.
    '''
    def __init__(self, data):
        self.data = data

    def get_trade_numbers(self):
        pos_data = self.data
        delta = pos_data - pos_data.shift(1)
        delta.iloc[0] = 0
        entries = (delta != 0) & (delta == pos_data)
        exits = (delta != 0) & (pos_data == 0)
        col_counts = (1 * entries).cumsum() * sign(pos_data)
        # We have trade counts starting from one in each column
        # Need to step each column up by the count from the previous
        col_step = col_counts.max().cumsum().shift(1)
        col_step.iloc[0] = 0
        # Before we add the step, need to replace zeroes with NaN
        # so that the zeroes don't get incremented
        col_counts[col_counts == 0] = NaN
        col_counts += col_step
        trades = Series(col_counts.T.values.flatten())
        return trades

    def get_trade_groups(self, data):
        '''
        Returns a groupy object of the supplied dataframe
        Note: We need to flatten the data into a Series to get a
        continuous trade count, otherwise it would be by column.
        '''
        trades = self.get_trade_numbers()
        return Series(data.T.values.flatten()).groupby(trades)



