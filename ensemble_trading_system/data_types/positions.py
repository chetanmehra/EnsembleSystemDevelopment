
from copy import deepcopy
from pandas import DataFrame
from numpy import sign

from system.interfaces import DataElement

from data_types.returns import StrategyReturns
from data_types.trades import Trade, TradeCollection
from data_types.events import EventCollection


class Position(DataElement):
    '''
    Position objects hold the dual role of keeping the position data for the strategy, as well as 
    calculating the returns from holding those positions.
    '''
    def __init__(self, data):
        if not isinstance(data, DataFrame):
            raise TypeError
        self.data = data
        self.events = EventCollection.from_position_data(data)

    def create_events(self):
        return EventCollection.from_position_data(self.data)

    def create_trades(self, strategy):
        prices = strategy.trade_prices
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
                exit = self.events.next_exit(entry, ticker_exits)
                trades.append(Trade(entry.ticker, entry.date, exit.date, 
                                    prices[entry.ticker], self.data[entry.ticker]))
        return TradeCollection(trades)


    def update_from_trades(self, trades):
        new_pos_data = deepcopy(self.data)
        new_pos_data[:] = 0
        for trade in trades.as_list():
            new_pos_data.loc[trade.entry:trade.exit, trade.ticker] = self.data.loc[trade.entry:trade.exit, trade.ticker]
        self.data = new_pos_data
        self.events = self.create_events()
        self.trades = trades


    def applied_to(self, market_returns):
        return StrategyReturns(self.data * market_returns.data, self)

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
            self.data[trade.ticker][trade.entry:trade.exit] = 0

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

