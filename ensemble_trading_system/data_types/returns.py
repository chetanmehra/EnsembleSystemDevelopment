'''
The returns module contains classes for dealing with different
types of returns data.
'''
from math import log
from pandas import DataFrame, Series, Categorical
from pandas.core.common import isnull
import matplotlib.pyplot as plt

from system.interfaces import DataElement
from system.metrics import Drawdowns
from data_types.constants import TRADING_DAYS_PER_YEAR



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

    def final(self):
        return self.log().sum()
    
    def cumulative(self):
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
        return (1 + self.final) ** (TRADING_DAYS_PER_YEAR / len(self)) - 1

    def drawdowns(self):
        return Drawdowns(self.cumulative())
        
    def sharpe(self):
        mean = self.data.mean()
        std = self.data.std()
        return (mean / std) * (TRADING_DAYS_PER_YEAR ** 0.5)

    def volatility(self):
        return self.data.std() * (TRADING_DAYS_PER_YEAR ** 0.5)

    def monthly(self):
        returns = DataFrame(self.data, index = self.data.index, columns = ["Returns"])
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



