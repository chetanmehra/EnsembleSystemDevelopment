'''
The returns module contains classes for dealing with different
types of returns data.
'''
from pandas import DataFrame, Series, Categorical
from pandas.core.common import isnull
from numpy import NaN, arange, log, exp
import matplotlib.pyplot as plt

from system.interfaces import DataElement
from system.metrics import Drawdowns, OptF
from data_types.constants import TRADING_DAYS_PER_YEAR


class Returns(DataElement):
    """
    Returns represents a returns Series, and provides methods for summarising
    and plotting.
    """
    def __init__(self, data):
        self.data = data.fillna(0)
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

    def int_indexed(self):
        '''
        Converts to integer based index.
        '''
        return Returns(Series(self.data.values))

    def final(self):
        '''
        Returns the final cumulative return of the series
        '''
        return exp(self.log().sum()) - 1
    
    def cumulative(self):
        returns = self.log()
        return exp(returns.cumsum()) - 1
    
    def log(self):
        returns = self.data
        returns[returns <= -1.0] = -0.9999999999
        return log(returns + 1)
        
    def plot(self, start = None, **kwargs):
        returns = self.cumulative()
        if start is not None:
            start_value = returns[start]
            if isinstance(start_value, Series):
                start_value = start_value[0]
            returns = returns - start_value
        returns[start:].plot(**kwargs)
        
    def annualised(self):
        return (1 + self.final()) ** (TRADING_DAYS_PER_YEAR / len(self)) - 1

    def drawdowns(self):
        return Drawdowns(self.cumulative())
        
    def sharpe(self):
        mean = self.data.mean()
        std = self.data.std()
        if isinstance(std, Series):
            std[std == 0] = NaN
        else:
            std = NaN
        return (mean / std) * (TRADING_DAYS_PER_YEAR ** 0.5)

    def optf(self):
        return OptF(self.data)

    def annual_mean(self):
        return self.data.mean() * TRADING_DAYS_PER_YEAR

    def volatility(self):
        return self.data.std() * (TRADING_DAYS_PER_YEAR ** 0.5)

    def normalised(self):
        volatility = self.volatility()
        if volatility == 0:
            return NaN
        else:
            return self.annualised() / volatility

    def monthly(self, start = None):
        '''
        monthly returns a dict with a number of dataframes of returns summarised
        by month (columns) and year (rows).
        monthly accepts a start parameter as a date to crop the data from.
        '''
        data = self.data[start:]
        returns = DataFrame(data.values, index = data.index, columns = ["Returns"])
        returns['Month'] = returns.index.strftime("%b")
        returns['Month'] = Categorical(returns['Month'], ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        returns['Year'] = returns.index.strftime("%Y")
        grouped = returns.groupby(["Year", "Month"])
        result = {}
        result["mean"] = grouped.mean().unstack() * 12
        result['std'] = grouped.std().unstack() * (12 ** 0.5)
        result['sharpe'] = result['mean'] / result['std']
        return result

    def plot_monthly(self, start = None, values = 'sharpe'):
        if isinstance(values, list):
            fig, axes = plt.subplots(nrows = 1, ncols = len(values))
        else:
            fig, ax = plt.subplots()
            axes = (ax, )
        
        results = self.monthly(start = start)

        for i, value in enumerate(values):
            data = results[value.lower()].transpose()
            title = value.upper()[0] + value.lower()[1:]
            x_labels = data.columns
            y_labels = data.index.levels[1]
            ax = axes[i]
            self.plot_heatmap(data.values, x_labels, y_labels, title, ax)
        
        fig.tight_layout()

    def plot_heatmap(self, values, x_labs, y_labs, title, ax):
        im = ax.imshow(values, cmap = "RdYlGn")
        ax.set_xticks(arange(len(x_labs)))
        ax.set_yticks(arange(len(y_labs)))
        ax.set_xticklabels(x_labs)
        ax.set_yticklabels(y_labs)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(y_labs)):
            for j in range(len(x_labs)):
                ax.text(j, i, round(values[i, j], 2), ha="center", va="center", fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax = ax)


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



