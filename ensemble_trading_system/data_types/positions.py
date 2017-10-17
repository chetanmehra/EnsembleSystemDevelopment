
from copy import deepcopy
from math import log
from pandas import DataFrame, Series, Categorical
from pandas.core.common import isnull
from numpy import sign
import matplotlib.pyplot as plt

from system.interfaces import DataElement
from system.metrics import Drawdowns
from data_types.trades import Trade, TradeCollection


class Position(DataElement):
    '''
    Position objects hold the dual role of keeping the position data for the strategy, as well as 
    calculating the returns from holding those positions.
    '''
    def __init__(self, data):
        if not isinstance(data, DataFrame):
            raise TypeError
        self.data = data
# Factory methods
    def create_trades(self, strategy):
        prices = strategy.get_trade_prices()
        trades = []
        position_sign = (self.data > 0) * 1
        position_sign[self.data < 0] = -1
        flags = position_sign - position_sign.shift(1)
        # clear missing values from first rows
        start_row = 0
        while all(flags.ix[start_row].isnull()):
            flags.ix[start_row] = 0
            start_row += 1
        # Add trade entries occuring on first day
        flags.ix[start_row][position_sign.ix[start_row] != 0] = position_sign.ix[start_row][position_sign.ix[start_row] != 0]
        for ticker in flags:
            ticker_flags = flags[ticker]
            ticker_sign = position_sign[ticker]
            # Flag values of -2 or 2 represents a complete switch from short to long or vice-versa.
            entries = ticker_flags.index[((ticker_flags != 0) & (ticker_sign != 0)) | (abs(ticker_flags) > 1)]
            exits = ticker_flags.index[((ticker_flags != 0) & (ticker_sign == 0)) | (abs(ticker_flags) > 1)]
            for entry_day in entries:
                valid_exits = (exits > entry_day)
                if not any(valid_exits):
                    exit_day = ticker_flags.index[-1]
                else:
                    exit_day = exits[valid_exits][0]
                trades.append(Trade(ticker, entry_day, exit_day, prices[ticker], self.data[ticker]))
        return TradeCollection(trades)

    def update_from_trades(self, trades):
        new_pos_data = deepcopy(self.data)
        new_pos_data[:] = 0
        for trade in trades.as_list():
            new_pos_data.loc[trade.entry:trade.exit, trade.ticker] = self.data.loc[trade.entry:trade.exit, trade.ticker]
        self.data = new_pos_data

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


class Returns(DataElement):
    """
    Returns represents a returns Series, and provides methods for summarising
    and plotting.
    """
    def __init__(self, data):
        data[isnull(data)] = 0
        self.data = data
        self.lag = 1
        self.calculation_timing = ["entry", "exit"]
        self.indexer = None
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)

    def __sub__(self, other):
        return Returns(self.data - other.data)
    
    def append(self, other):
        self.data[other.data.columns] = other.data
    
    def cumulative(self):
        returns = self.data
        returns = self.log()
        return returns.cumsum()
    
    def log(self):
        returns = self.data
        returns[returns <= -1.0] = -0.9999999999
        return (returns + 1).apply(log)
        
    def plot(self, start = None, **kwargs):
        returns = self.cumulative()
        if start is not None:
            start_value = returns[start]
            if isinstance(start_value, Series):
                start_value = start_value[0]
            returns = returns - start_value
        returns[start:].plot(**kwargs)
        
    def annualised(self):
        returns = self.data
        return (1 + returns) ** 260 - 1

    def drawdowns(self):
        return Drawdowns(self.cumulative())
        
    def sharpe(self):
        returns = self.annualised()
        mean = returns.mean()
        std = returns.std()
        return mean / std

    def monthly(self):
        returns = DataFrame(self.data, columns = ["Returns"])
        returns['Month'] = returns.index.strftime("%b")
        returns['Month'] = Categorical(returns['Month'], ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        returns['Year'] = returns.index.strftime("%Y")
        grouped = returns.groupby(["Year", "Month"])
        result = {}
        result["mean"] = grouped.mean().unstack()
        result['std'] = grouped.std().unstack()
        result['sharpe'] = result['mean'] / result['std']
        return result

    def plot_monthly(self, values = 'sharpe'):
        returns = self.monthly()[values.lower()]
        title = values.upper()[0] + values.lower()[1:]
        ax = plt.imshow(returns.values, cmap = "RdYlGn")
        ax.axes.set_xticklabels(returns.columns.levels[1])
        ax.axes.set_yticklabels(returns.index)
        plt.colorbar()
        plt.title(title)
        return ax
    
    
class AggregateReturns(Returns):
    """
    AggregateReturns represents a dataframe of returns which are summed for the overall
    result (e.g. the result of position weighted returns).
    """
    def __init__(self, data):
        super().__init__(data)
        self.columns = data.columns

    def combined(self):
        return Returns(self.data.sum(axis = 1))

    def log(self):
        return self.combined().log()

    def annualised(self):
        return self.combined().annualised()
    
    
class AverageReturns(Returns):
    """
    AverageReturns represents a dataframe of returns which are averaged for the overal
    result (e.g. market returns).
    """
    def __init__(self, data):
        super().__init__(data)
        self.columns = data.columns
    
    def combined(self):
        return Returns(self.data.mean(axis = 1))

    def log(self):
        return self.combined().log()

    def annualised(self):
        return self.combined().annualised()

class StrategyReturns(AggregateReturns):
    """
    StrategyReturns is a hypothetical result of the strategy assuming that rebalancing
    can happen daily at no cost, to ensure that the strategy is fully invested. Positions 
    are divided by the number of concurrent positions for the day to normalise.
    """
    def __init__(self, data, positions):
        data = data.div(positions.num_concurrent(), axis = 'rows')
        super().__init__(data)



