
from pandas import Panel, DataFrame, Series
from pandas.core.common import isnull
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import sign
from math import log

from System.Strategy import StrategyContainerElement, PositionSelectionElement


class Position(StrategyContainerElement):
    
    def __init__(self, data):
        if type(data) is not DataFrame:
            raise TypeError
        self.data = data
        
    def long_only(self):
        data = deepcopy(self.data)
        data[data < 0] = 0
        return Position(data)
        
    def short_only(self):
        data = deepcopy(self.data)
        data[data > 0] = 0
        return Position(data)

    def remove(self, trade):
        self.data[trade.ticker][trade.entry:trade.exit] = 0
        
    def applied_to(self, returns):
        return AggregateReturns(self.data * returns.data)


class Filter(object):

    def __init__(self, values):
        '''
        Requires that values is a DataFrame with first column 'ticker' and next column filter values.
        '''
        self.values = {}
        self.name = values.columns[-1]
        for ticker in set(values["ticker"]):
            self.values[ticker] = values[values["ticker"] == ticker].iloc[:, -1]

    def at(self, date, ticker):
        ticker_values = self.values[ticker]
        values = ticker_values[ticker_values.index <= date]
        if len(values):
            return values.iloc[-1]
        else:
            return None


    def __call__(self, strategy, range):
        '''
        Removes position values where filter criteria is not met.
        range should be a tuple representing the acceptable filter range.
        '''
        left = min(range)
        right = max(range)
        condition = lambda trade: trade.filter(self, "entry") < left or trade.filter(self, "entry") > right
        eliminated = strategy.trades.filter(condition)
        for trade in eliminated.trades:
            strategy.positions.remove(trade)




class Trade(object):

    def __init__(self, ticker, prices, entry_date, exit_date):
        self.ticker = ticker
        self.prices = prices
        self.entry = entry_date
        self.exit = exit_date
        self.duration = (exit_date - entry_date).days
        self.normalised = Series((prices[self.entry:self.exit] / prices[self.entry]).values) - 1

    def plot_normalised(self):
        self.normalised.plot()

    @property
    def price_at_entry(self):
        return self.prices[self.entry]

    @property
    def price_at_exit(self):
        return self.prices[self.exit]

    @property
    def base_return(self):
        return self.normalised.iloc[-1]

    @property
    def annualised_return(self):
        return (sum(self.normalised.apply(log))) ** (260 / self.duration) - 1

    @property
    def MAE(self):
        return min(self.normalised)

    @property
    def MFE(self):
        return max(self.normalised)

    def filter(self, filter, timing):
        '''
        Gets the filter value vs price at specified timing (entry/exit)
        '''
        if timing not in ["entry", "exit"]:
            raise ValueError("Timing must be entry or exit")
        date = getattr(self, timing)
        price = getattr(self, "price_at_" + timing)
        filter_value = filter.at(date, self.ticker)
        if filter_value is not None:
            return filter_value / price
        else:
            return None
    


class TradeCollection(object):

    def __init__(self, data):
        '''
        TradeCollection supports two ways of construction.
        1. From a list of trades - e.g. from filtering an existing collection.
        2. From a Strategy object - in which case it will create from scratch.
        '''
        if isinstance(data, list):
            self.tickers = list(set(trade.ticker for trade in data))
            self.trades = data
        else:
            self.tickers = data.market.tickers
            self.trades = []
            for ticker in self.tickers:
                self.trades.extend(self.create_trades(data.positions, ticker, data.market.close[ticker]))
        self.plot_series = TradePlotSeriesCollection()
    

    def create_trades(self, positions, ticker, prices):

        flags = positions.data - positions.data.shift(1)
        entries = flags.index[flags[ticker] > 0]
        trades = []
        for i in range(1, len(entries)):
            entry = entries[i - 1]
            next = entries[i]
            exit = flags[entry:next].index[flags[ticker][entry:next] < 0][0]
            if exit is None:
                exit = flags.index[-1]
            trades.append(Trade(ticker, prices, entry, exit))
        return trades

    def __getitem__(self, key):
        return filter(lambda trade:trade.ticker == key, self.trades)

    @property
    def returns(self):
        return [trade.base_return for trade in self.trades]

    @property
    def mean_return(self):
        return Series(self.returns).mean()

    @property
    def std_return(self):
        return Series(self.returns).std()

    @property
    def Sharpe(self):
        returns = Series(self.returns)
        return returns.mean() / returns.std()

    @property
    def G(self):
        # Refer research note: EQ-2011-003
        S_sqd = self.Sharpe ** 2
        return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1


    def max_duration(self):
        return max([trade.duration for trade in self.trades])

    def max_MAE(self):
        return min([trade.MAE for trade in self.trades])

    def max_MFE(self):
        return max([trade.MFE for trade in self.trades])

    def filter(self, condition):
        '''
        filter accepts a lambda expression which must accept a Trade object as its input.
        A dictionary of list of trades meeting the condition is returned.
        '''
        sub_trades = filter(condition, self.trades)
        return TradeCollection(sub_trades)
        
    def clear_plot_series(self):
        self.plot_series = TradePlotSeriesCollection()

    def create_plot_series(self, filter_object, boundaries, summary_method):
        '''
        Creates a series for plotting trade return measures vs a filter, e.g. the mean
        returns for a range of filter values.
        Inputs:
            filter_object - the filter data as a Filter object
            boundaries - list of end points for each of the buckets, plot series point will 
                   be created at the mid point of each bucket.
            summary_method - the method to be applied to the subset of trade returns. must be one of the
                   methods on TradeCollection e.g. mean, std, Sharpe, G.
        Outputs:
            PlotSeries is added to the TradeCollection for plotting.
        '''
        partition_labels = []
        results = []
        for i in range(0, len(boundaries) - 1):
            left = boundaries[i]
            right = boundaries[i + 1]
            partition_labels.append("[{0}, {1})".format(left, right))
            condition = lambda trade:((trade.filter(filter_object, "entry") >= left) and (trade.filter(filter_object, "entry") < right))
            trades  = TradeCollection(filter(condition, list(self.trades)))
            results.append(getattr(trades, summary_method))
        self.plot_series.append(Series(results, partition_labels, name = ": ".join([summary_method, filter_object.name])))


    def create_filter_summary(self, filter_values, boundaries, timing = "entry"):
        '''
        filter_values is assumed to be a dataframe with first column 'ticker' and remaining columns as different, 
        but related filter values.
        '''
        if timing not in ["entry", "exit"]:
            raise ValueError("timing muse be entry or exit")

        filter_types = filter_values.columns[1:]
        boundary_tuples = zip(boundaries[:-1], boundaries[1:])
        partition_labels = ["[{0}, {1})".format(left, right) for left, right in boundary_tuples]
        filter_summary = Panel(None, ["mean", "std"], partition_labels, filter_types)

        for type in filter_types:
            filter_object = Filter(filter_values[["ticker", type]])
            mean = []
            std_dev = []
            for left, right in boundary_tuples:
                condition = lambda trade:((trade.filter(filter_object, "entry") >= left) and (trade.filter(filter_object, "entry") < right))
                trades = TradeCollection(filter(condition, list(self.trades)))
                mean.append(trades.mean_return)
                std_dev.append(trades.std_return)
            filter_summary["mean"][type] = mean
            filter_summary["std"][type] = std_dev

        return filter_summary


    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_normalised()


    def return_vs_filter(self, filter):
        '''
        Filter needs to be entered as a pandas.DataFrame.
        '''
        returns, filters = zip(*[(trade.filter(filter, "entry"), trade.base_return) for trade in self.trades])
        return (filters, returns)


    def plot_return_vs_filter(self, filter, xlim = None):
        plt.scatter(*self.return_vs_filter(filter))
        plt.plot([-10, 10], [0, 0], "k-")
        if xlim is not None:
            plt.xlim(xlim)


class TradePlotSeriesCollection(object):

    def __init__(self, color_map = "jet"):
        self.color_map = color_map
        self.collection = DataFrame()

    def plot(self, **kwargs):
        self.collection.plot()

    def append(self, series):
        self.collection[series.name] = series


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
        
    def plot(self, collapse_fun, **kwargs):
        returns = self.cumulative(collapse_fun)
        returns.plot(**kwargs)
        
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
    
    
class AverageReturns(Returns):
    
    def annualised(self):
        returns = self.collapse_by("mean")
        return (1 + returns) ** 260 - 1
        

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
  



