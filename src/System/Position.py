
from pandas.core.frame import DataFrame, Series
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
        
    def applied_to(self, returns):
        return AggregateReturns(self.data * returns.data)


class Trade(object):

    def __init__(self, prices, entry_date, exit_date):
        self.prices = prices
        self.entry = entry_date
        self.exit = exit_date
        self.duration = (exit_date - entry_date).days
        self.normalised = Series((prices[self.entry:self.exit] / prices[self.entry]).values)

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
        return (1 + sum(self.normalised.apply(log))) ** (260 / self.duration) - 1

    @property
    def MAE(self):
        return min(self.normalised)

    @property
    def MFE(self):
        return max(self.normalised)

    def filter(self, filter, timing):
        '''
        Gets the filter value at specified timing (entry/exit)
        Assumes filter is for the same ticker as trade.
        '''
        if timing not in ["entry", "exit"]:
            raise ValueError("Timing must be entry or exit")
        date = getattr(self, timing)
        price = getattr(self, "price_at_" + timing)
        filter_value = filter[filter.index <= date]
        if len(filter_value):
            return filter_value.iloc[-1] / price
        else:
            return None
    


class TradeCollection(object):

    def __init__(self, data):
        '''
        TradeCollection supports two ways of construction.
        1. From a dictionary of trades {ticker:[Trades]} - e.g. from filtering an existing collection.
        2. From a Strategy object - in which case it will create from scratch.
        '''
        if isinstance(data, dict):
            self.tickers = data.keys()
            self.trades = data
        else:
            self.tickers = data.market.tickers
            self.trades = {}
            for ticker in self.tickers:
                self.trades[ticker] = self.create_trades(data.positions, ticker, data.market[ticker]["Close"])
        

    def create_trades(self, positions, ticker, prices):
        '''
        Note: this requires positions to be all normalised to 1.
        This could be done with sign(positions).
        '''
        flags = positions.data - positions.data.shift(1)
        entries = flags.index[flags[ticker] == 1]
        trades = []
        for i in range(1, len(entries)):
            entry = entries[i - 1]
            next = entries[i]
            exit = flags[entry:next].index[flags[ticker][entry:next] == -1][0]
            if exit is None:
                exit = flags.index[-1]
            trades.append(Trade(prices, entry, exit))
        return trades

    def __getitem__(self, key):
        return self.trades[key]

    @property
    def returns(self):
        all_returns = []
        for ticker in self.trades:
            all_returns.extend([trade.base_return for trade in self.trades[ticker]])
        return all_returns

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
        overall_max = 0
        for ticker in self.trades:
            set_max = max([trade.duration for trade in self.trades[ticker]])
            overall_max = max([overall_max, set_max])
        return overall_max

    def max_MAE(self):
        overall_MAE = 0
        for ticker in self.trades:
            set_min = min([trade.MAE for trade in self.trades[ticker]])
            overall_MAE = min([overall_MAE, set_min])
        return overall_MAE

    def max_MFE(self):
        overall_MFE = 0
        for ticker in self.trades:
            set_max = max([trade.MFE for trade in self.trades[ticker]])
            overall_MFE = max([overall_MFE, set_max])
        return overall_MFE

    def filter(self, condition):
        '''
        filter accepts a lambda expression which must accept a Trade object as its input.
        A dictionary of list of trades meeting the condition is returned.
        '''
        sub_trades = {ticker:filter(condition, self.trades[ticker]) for ticker in self.tickers}
        sub_trades = {ticker:trades for ticker, trades in sub_trades.items() if len(trades) > 0}
        return TradeCollection(sub_trades)


    def create_plot_series(self, filter_frame, boundaries, summary_method):
        '''
        Creates a series for plotting trade return measures vs a filter, e.g. the mean
        returns for a range of filter values.
        Inputs:
            filter_frame - the filter data as a pandas.DataFrame
            boundaries - list of end points for each of the buckets, plot series point will 
                   be created at the mid point of each bucket.
            summary_method - the method to be applied to the subset of trade returns. must be one of the
                   methods on TradeCollection e.g. mean, std, Sharpe, G.
        Outputs:
            PlotSeries is added to the TradeCollection for plotting.
        '''
        partition_centres = []
        results = []
        self.plot_series = TradePlotSeriesCollection()

        for i in range(0, len(boundaries) - 1):
            left = boundaries[i]
            right = boundaries[i + 1]
            partition_centres.append((left + right) / 2.0)

            sub_trades = {}
            for ticker in self.trades:
                filter_values = filter_frame[filter_frame["ticker"] == ticker]["Base"]
                condition = lambda trade: ((trade.filter(filter_values, "entry") >= left) and (trade.filter(filter_values, "entry") <= right))
                filtered_trades = filter(condition, self.trades[ticker])
                if len(filtered_trades):
                    sub_trades[ticker] = filtered_trades
            trades  = TradeCollection(sub_trades)
            results.append(getattr(trades, summary_method))
        self.plot_series.append(TradePlotSeries(partition_centres, results))


    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_normalised()


    def return_vs_filter(self, filter):
        '''
        Filter needs to be entered as a pandas.DataFrame.
        '''
        returns = []
        filters = []
        for ticker in self.tickers:
            base_value = filter["Base"].loc[filter["ticker"] == ticker]
            for trade in self[ticker]:
                try:
                    filter_at_entry = base_value[base_value.index < trade.entry][-1]
                    filter_at_entry = filter_at_entry / trade.price_at_entry
                    filters.append(filter_at_entry)
                except IndexError:
                    filters.append(None)
                
                returns.append(trade.base_return)
        return (filters, returns)


    def plot_return_vs_filter(self, filter):
        plt.scatter(*self.return_vs_filter(filter))


class TradePlotSeries(object):

    def __init__(self, x, y, col = "red", style = "-"):
        self.x = x
        self.y = y
        self.col = col
        self.style = style

    def plot(self):
        plt.plot(self.x, self.y, color = self.col, ls = self.style)


class TradePlotSeriesCollection(object):

    def __init__(self, color_map = "jet"):
        self.color_map = color_map
        self.plot_series = []

    def plot(self, **kwargs):
        color_map = plt.get_cmap(self.color_map)
        col_index = 80
        col_increment = round(60 / len(self.plot_series))
        for series in self.plot_series:
            series.col = color_map(col_index)
            series.plot(**kwargs)
            col_index += col_increment

    def append(self, series):
        self.plot_series.append(series)


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
  



