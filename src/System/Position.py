
from pandas import Panel, DataFrame, Series
from pandas.core.common import isnull
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import sign
from math import log

from System.Strategy import StrategyContainerElement, PositionSelectionElement
from System.Trade import TradeCollection, Trade


class Position(StrategyContainerElement):
    
    def __init__(self, data):
        if type(data) is not DataFrame:
            raise TypeError
        self.data = data
        self.trades = None


    def create_trades(self, entry_prices, exit_prices):
        trades = []
        flags = self.data - self.shift(1).data
        for ticker in self.tickers:
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
        self.trades = TradeCollection(trades, self.tickers)


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
        
    def applied_to(self, returns):
        return AggregateReturns(self.data * returns.data)

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

    def filter_summary(self, filter_values, boundaries):
        return self.trades.filter_summary(filter_values, boundaries)

    def filter_comparison(self, filter_values, filter1_type, filter2_type, boundaries1, boundaries2):
        return self.trades.filter_comparison(filter_values, filter1_type, filter2_type, boundaries1, boundaries2)



class FilteredPositions(Position):

    def __init__(self, original_positions, new_positions, new_trades, new_tickers):
        self.original_positions = original_positions
        self.data = new_positions
        self.trades = TradeCollection(new_trades, new_tickers)


    def applied_to(self, returns):
        return FilteredReturns(self.normalised().applied_to(returns), 
                               self.original_positions.normalised().applied_to(returns))


    def long_only(self):
        long_positions = self.copy()
        long_positions.data[long_positions.data < 0] = 0
        long_positions.original_positions = long_positions.original_positions.long_only()
        return long_positions

    def short_only(self):
        short_positions = self.copy()
        short_positions.data[short_positions.data > 0] = 0
        short_positions.original_positions = short_positions.original_positions.short_only()
        return short_positions


    def filter_summary(self, filter_values, boundaries):
        return self.original_positions.filter_summary(filter_values, boundaries)




class Returns(object):
    
    def __init__(self, data):
        data[isnull(data)] = 0
        self.data = data
        self.columns = data.columns
        
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
        return (returns + 1).apply(log)
        
    def plot(self, collapse_fun, start, **kwargs):
        returns = self.cumulative(collapse_fun)
        if start is not None:
            returns = returns - returns[start]
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

    def plot(self, start, **kwargs):
        super(AggregateReturns, self).plot("sum", start, **kwargs)
    
    
class AverageReturns(Returns):
    
    def annualised(self):
        returns = self.collapse_by("mean")
        return (1 + returns) ** 260 - 1

    def plot(self, start, **kwargs):
        return super(AverageReturns, self).plot("mean", start, **kwargs)


class FilteredReturns(Returns):

    def __init__(self, main, other):
        self.main_returns = main
        self.comparison_returns = other


    def plot(self, **kwargs):
        self.main_returns.plot(label = "Filtered", **kwargs)
        if "color" in kwargs:
            specified_color = kwargs.pop("color")
            if specified_color == "red":
                color = "purple"
        self.comparison_returns.plot(color = "red", label = "Base", **kwargs)

    def annualised(self):
        return self.main_returns.annualised()
        

        

class DefaultPositions(PositionSelectionElement):

    def execute(self, strategy):
        optimal_size = strategy.forecasts.optF()
        return Position(sign(optimal_size))


class SingleLargestF(PositionSelectionElement):
    
    def execute(self, strategy):
        optimal_size = strategy.forecasts.optF()
        result = deepcopy(optimal_size)
        result[:] = 0
        maximum_locations = optimal_size.abs().idxmax(axis = 1)
        directions = sign(optimal_size)
        
        for row, col in enumerate(maximum_locations):
            if col not in result.columns:
                pass
            else:
                result[col][row] = directions[col][row] 
        
        return Position(result)
    
    
class HighestRankedFs(PositionSelectionElement):
    
    def __init__(self, num_positions):
        self.num_positions = num_positions
        
    def execute(self, strategy):
        optimal_size = strategy.forecasts.optF()
        result = deepcopy(optimal_size)
        result[:] = 0
        ranks = abs(optimal_size).rank(axis = 1, ascending = False)
        result[ranks <= self.num_positions] = 1
        result *= sign(optimal_size)
        result /= self.num_positions
        return Position(result)
  



