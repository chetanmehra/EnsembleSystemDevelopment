
from pandas import Panel, DataFrame, Series
from pandas.core.common import isnull
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import sign
from math import log

from System.Strategy import DataElement
from System.Signal import Signal


class Position(DataElement):
    '''
    Position objects hold the dual role of keeping the position data for the strategy, as well as 
    calculating the returns from holding those positions.
    '''
    def __init__(self, data):
        if type(data) is not DataFrame:
            raise TypeError
        self.data = data


    def updateFromTrades(self, trades):
        new_pos_data = deepcopy(self.data)
        new_pos_data[:] = 0
        for trade in trades.as_list():
            new_pos_data.loc[trade.entry:trade.exit, trade.ticker] = self.data.loc[trade.entry:trade.exit, trade.ticker]
        self.data = new_pos_data


    def appliedTo(self, market_returns):
        return AggregateReturns(self.data * market_returns.data, market_returns.indexer)


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
    
    def __init__(self, data, indexer):
        data[isnull(data)] = 0
        self.data = data
        self.columns = data.columns
        self.lag = 1
        self.calculation_timing = ["entry", "exit"]
        self.indexer = indexer
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
    def append(self, other):
        self.data[other.data.columns] = other.data
    
    def collapse_by(self, function):
        allowable_functions = ["mean", "sum"]
        if function not in allowable_functions:
            raise TypeError("function must be one of: " + ", ".join(allowable_functions))
        return self.data.__getattribute__(function)(axis = 1)
        
    def cumulative(self, collapse_fun):
        returns = self.log(collapse_fun)
        return returns.cumsum()
    
    def log(self, collapse_fun):
        returns = self.collapse_by(collapse_fun)
        returns[returns <= -1.0] = -0.9999999999
        return (returns + 1).apply(log)
        
    def plot(self, collapse_fun, start = None, **kwargs):
        returns = self.cumulative(collapse_fun)
        if start is not None:
            start_value = returns[start]
            if isinstance(start_value, Series):
                start_value = start_value[0]
            returns = returns - start_value
        returns[start:].plot(**kwargs)
        
    def annualised(self, collapse_fun):
        returns = self.collapse_by(collapse_fun)
        return (1 + returns) ** 260 - 1
        
    def sharpe(self, collapse_fun):
        returns = self.annualised(collapse_fun)
        mean = returns.mean()
        std = returns.std()
        return mean / std
    
    
class AggregateReturns(Returns):
    
    def annualised(self):
        returns = self.collapse_by("sum")
        return (1 + returns) ** 260 - 1

    def plot(self, start = None, **kwargs):
        super(AggregateReturns, self).plot("sum", start, **kwargs)
    
    
class AverageReturns(Returns):
    
    def annualised(self):
        returns = self.collapse_by("mean")
        return (1 + returns) ** 260 - 1

    def plot(self, start = None, **kwargs):
        return super(AverageReturns, self).plot("mean", start, **kwargs)

        



