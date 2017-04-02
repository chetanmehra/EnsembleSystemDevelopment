
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


class TradeCollection(object):

    def __init__(self, data, tickers):
        self.tickers = tickers
        self.trades = data

    def __getitem__(self, key):
        return [trade for trade in self.trades if trade.ticker == key]

    def as_list(self):
        return self.trades

    def as_dataframe(self):
        data = [trade.as_tuple() for trade in self.trades]
        return DataFrame(data, columns = self.trades[0].cols)

    def as_dataframe_with(self, values):
        '''
        requires that values is a dataframe with first column ticker
        and remaining columns values to be inserted.
        Assumes index of values are dates / timestamps.
        '''
        df = self.as_dataframe()
        cols = values.columns[1:]
        for col in cols:
            df[col] = None

        if len(cols) == 1:
            cols = cols[0]

        for i in df.index:
            tick_values = values[values.ticker == df.ticker[i]][cols]
            existing_values = tick_values[tick_values.index < df.entry[i]]
            if not existing_values.empty:
                df.loc[i, cols] = existing_values.iloc[-1]

        return df


    @property
    def returns(self):
        return [trade.base_return for trade in self.trades]

    @property
    def durations(self):
        return [trade.duration for trade in self.trades]

    @property
    def MAEs(self):
        return [trade.MAE for trade in self.trades]

    @property
    def MFEs(self):
        return [trade.MFE for trade in self.trades]

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

    @property
    def max_duration(self):
        return max(self.durations)

    @property
    def max_MAE(self):
        return min(self.MAEs)

    @property
    def max_MFE(self):
        return max(self.MFEs)


    def find(self, condition):
        '''
        find accepts a lambda expression which must accept a Trade object as its input and return True/False.
        A TradeCollection of trades meeting the condition is returned.
        '''
        trades = [trade for trade in self.trades if condition(trade)]
        tickers = list(set([trade.ticker for ticker in trades]))
        return TradeCollection(trades, tickers)
        

    def filter_summary(self, filter_values, boundaries):
        '''
        filter_values is assumed to be a dataframe with first column 'ticker' and remaining columns as different, 
        but related filter values.
        boundaries is an iterable of boundary points e.g. (-1, 0, 0.5, 1, etc...)
        '''
        filter_types = filter_values.columns[1:]
        boundary_tuples = zip(boundaries[:-1], boundaries[1:])
        partition_labels = ["[{0}, {1})".format(left, right) for left, right in boundary_tuples]
        trade_df = self.as_dataframe_with(filter_values)
        mu = DataFrame(data = None, index = partition_labels, columns = filter_types)
        sd = DataFrame(data = None, index = partition_labels, columns = filter_types)
        N = DataFrame(data = None, index = partition_labels, columns = filter_types)
        for type in filter_types:
            ratio = trade_df[type] / trade_df.entry_price
            for i, bounds in enumerate(boundary_tuples):
                filtered = trade_df.base_return[(ratio >= bounds[0]) & (ratio < bounds[1])]
                mu.loc[partition_labels[i], type] = filtered.mean()
                sd.loc[partition_labels[i], type] = filtered.std()
                N.loc[partition_labels[i], type] = filtered.count()
        return {"mean" : mu, "std" : sd, "count" : N}


    def filter_comparison(self, filter_values, filter1_type, filter2_type, boundaries1, boundaries2):
        '''
        Provides a matrix comparing mean, std dev, and count for each combination of filter
        values. Note each filter should have only one column. It is assumed that the last 
        column of the filter value contains the filter of interest.
        '''

        boundary_tuples1 = zip(boundaries1[:-1], boundaries1[1:])
        boundary_tuples2 = zip(boundaries2[:-1], boundaries2[1:])
        partition_labels1 = ["[{0}, {1})".format(left, right) for left, right in boundary_tuples1]
        partition_labels2 = ["[{0}, {1})".format(left, right) for left, right in boundary_tuples2]
        trade_df = self.as_dataframe_with(filter_values[["ticker", filter1_type, filter2_type]])
        mu = DataFrame(data = None, index = partition_labels1, columns = partition_labels2)
        mu.index.name = filter1_type
        mu.columns.name = filter2_type
        sd = DataFrame(data = None, index = partition_labels1, columns = partition_labels2)
        sd.index.name = filter1_type
        sd.columns.name = filter2_type
        N = DataFrame(data = None, index = partition_labels1, columns = partition_labels2)
        N.index.name = filter1_type
        N.columns.name = filter2_type
        ratio1 = trade_df[filter1_type] / trade_df.entry_price
        ratio2 = trade_df[filter2_type] / trade_df.entry_price
        for i, bounds1 in enumerate(boundary_tuples1):
            first_filter = (ratio1 >= bounds1[0]) & (ratio1 < bounds1[1])
            for j, bounds2 in enumerate(boundary_tuples2):
                second_filter = (ratio2 >= bounds2[0]) & (ratio2 < bounds2[1])
                filtered = trade_df.base_return[first_filter & second_filter]
                mu.loc[partition_labels1[i], partition_labels2[j]] = filtered.mean()
                sd.loc[partition_labels1[i], partition_labels2[j]] = filtered.std()
                N.loc[partition_labels1[i], partition_labels2[j]] = filtered.count()
        return {"mean" : mu, "std" : sd, "count" : N}


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
        plt.plot([0, 0], [-10, 10], "k--")
        plt.plot([1, 1], [-10, 10], "k--")
        if xlim is not None:
            plt.xlim(xlim)


class Trade(object):

    def __init__(self, ticker, entry_date, exit_date, entry_prices, exit_prices):
        self.ticker = ticker
        self.entry = entry_date
        self.exit = exit_date
        self.entry_price = self.get_price(entry_date, entry_prices)
        self.exit_price = self.get_price(exit_date, exit_prices)
        self.duration = (exit_date - entry_date).days
        self.normalised = Series((exit_prices[self.entry:self.exit] / self.entry_price).values) - 1
        self.cols = ["ticker", "entry", "exit", "entry_price", "exit_price", "base_return", "duration"]


    def get_price(self, date, prices):
        price = prices[date]
        #if isnull(price):
        #    price = prices[prices.index >= date].dropna()[0]
        return price

    def plot_normalised(self):
        self.normalised.plot()

    def as_tuple(self):
        return tuple(getattr(self, name) for name in self.cols)

    @property
    def base_return(self):
        return (self.exit_price / self.entry_price) - 1

    @property
    def annualised_return(self):
        return (sum(self.normalised.apply(log))) ** (260 / self.duration) - 1

    @property
    def MAE(self):
        return min(self.normalised)

    @property
    def MFE(self):
        return max(self.normalised)



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



class Filter(PositionSelectionElement):

    def __init__(self, values, filter_range):
        '''
        Requires that values is a DataFrame with first column 'ticker' and next column filter values.
        filter_range should be a tuple representing the acceptable filter range.
        '''
        self.values = values
        self.bounds = filter_range
        self.left = min(filter_range)
        self.right = max(filter_range)
        self.name = values.columns[-1]


    def execute(self, strategy):
        '''
        Removes position values where filter criteria is not met.
        '''
        trades = strategy.positions.trades.as_list()
        trade_frame = strategy.positions.trades.as_dataframe_with(self.values)
        trade_frame["Filter"] = trade_frame[self.name] / trade_frame.entry_price
        filtered_bools = (trade_frame.Filter > self.left) & (trade_frame.Filter <= self.right)

        accepted_trades = []
        eliminated_trades = []

        for i, keep in enumerate(filtered_bools.values):
            if keep:
                accepted_trades.append(trades[i])
            else:
                eliminated_trades.append(trades[i])

        new_positions = strategy.positions.copy()
        new_positions.remove(eliminated_trades)

        return FilteredPositions(strategy.positions, new_positions.data, accepted_trades, new_positions.tickers)


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
  



