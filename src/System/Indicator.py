'''
Created on 13 Dec 2014

@author: Mark
'''
from System.Strategy import StrategyContainerElement
from System.Trade import TradeCollection, Trade

class Indicator(StrategyContainerElement):
    '''
    Indicator represent possible trades for the strategy.
    data should be a dataframe of True / False values for determine trade entries & exits
    measures should be a panel of the meaures used to derive the signals.
    '''

    def __init__(self, data, measures):
        self.data = data
        self.measures = measures
    
    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns
    
    def __getitem__(self, ticker):
        return self.data[ticker]
    
    def __len__(self):
        return len(self.data)

    def create_trades(self, strategy):
        ind = strategy.lagged_indicator
        entry_prices = strategy.get_entry_prices()
        exit_prices = strategy.get_exit_prices()
        trades = []
        flags = ind.data - ind.data.shift(1)
        flags.ix[0] = 0
        flags.ix[0][ind.data.ix[0] != 0] = 1
        for ticker in flags:
            ticker_flags = flags[ticker]
            entries = ticker_flags.index[ticker_flags > 0]
            i = 0
            while i < len(entries):
                entry = entries[i]
                i += 1
                if i < len(entries):
                    next = entries[i]
                else:
                    next = None
                exit = ticker_flags[entry:next].index[ticker_flags[entry:next] < 0]
                if len(exit) == 0:
                    exit = ticker_flags.index[-1]
                else:
                    exit = exit[0]
                trades.append(Trade(ticker, entry, exit, entry_prices[ticker], exit_prices[ticker]))
        return TradeCollection(trades)

    def plot_measures(self, ticker, start = None, end = None, ax = None):
        self.measures.minor_xs(ticker)[start:end].plot(ax = ax)



class LevelIndicator(Indicator):

    def __init__(self, data, measures):
        super().__init__(data, measures)
        self._levels = None

    @property
    def levels(self):
        if self._levels is None:
            for ticker in self.data:
                data = self.data[ticker]
                data = data[notnull(data)]
                self._levels = set(data)
            self._levels = sorted(self._levels)
        return self._levels




