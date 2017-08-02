
from copy import deepcopy
from math import log
from pandas import DataFrame, Series
from pandas.core.common import isnull
from numpy import sign

from system.interfaces import DataElement
from system.metrics import Drawdowns


class Position(DataElement):
    '''
    Position objects hold the dual role of keeping the position data for the strategy, as well as 
    calculating the returns from holding those positions.
    '''
    def __init__(self, data):
        if not isinstance(data, DataFrame):
            raise TypeError
        self.data = data

    def update_from_trades(self, trades):
        new_pos_data = deepcopy(self.data)
        new_pos_data[:] = 0
        for trade in trades.as_list():
            new_pos_data.loc[trade.entry:trade.exit, trade.ticker] = self.data.loc[trade.entry:trade.exit, trade.ticker]
        self.data = new_pos_data

    def applied_to(self, market_returns):
        return AverageReturns(self.data * market_returns.data, market_returns.indexer)

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
        data = deepcopy(self.data)
        data = data.div(self.num_concurrent(), axis = 0)
        return Position(data)


class Returns(DataElement):
    """
    Returns represents a returns Series, and provides methods for summarising
    and plotting.
    """
    def __init__(self, data, indexer):
        data[isnull(data)] = 0
        self.data = data
        self.lag = 1
        self.calculation_timing = ["entry", "exit"]
        self.indexer = indexer
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
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
        return (1 + returns) ** 260 - 1

    def drawdowns(self):
        return Drawdowns(self.cumulative())
        
    def sharpe(self):
        returns = self.annualised()
        mean = returns.mean()
        std = returns.std()
        return mean / std
    
    
class AggregateReturns(Returns):
    """
    AggregateReturns represents a dataframe of returns which are summed for the overall
    result (e.g. the result of position weighted returns).
    """
    def __init__(self, data, indexer):
        super().__init__(data, indexer)
        self.columns = data.columns

    def log(self):
        return Returns(self.data.sum(axis = 1), self.indexer).log()

    def annualised(self):
        return Returns(self.data.sum(axis = 1), self.indexer).annualised()
    
    
class AverageReturns(Returns):
    """
    AverageReturns represents a dataframe of returns which are averaged for the overal
    result (e.g. market returns).
    """
    def __init__(self, data, indexer):
        super().__init__(data, indexer)
        self.columns = data.columns
    
    def log(self):
        return Returns(self.data.mean(axis = 1), self.indexer).log()

    def annualised(self):
        return Returns(self.data.mean(axis = 1), self.indexer).annualised()



