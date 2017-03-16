'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from pandas import Panel, DataFrame, Series


class Strategy(object):
    '''
    Strategy holds the components of a trading strategy, including indicators, forecasts etc.
    It manages the creation components as well as the resulting data. 
    '''

    def __init__(self, trade_timing, ind_timing):
        self.indexer = Indexer(trade_timing, ind_timing)
        self.required_fields = ["market", "measure", "model", "select_positions"]
        # Container elements
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.filtered_positions = None
        self.trades = None
        # Strategy creation elements
        self.measure = None
        self.model = None
        self.select_positions = None
        self.filter = None
        
    def check_fields(self):
        missing_fields = []
        for field in self.required_fields:
            try:
                self.__getattribute__(field)
            except AttributeError:
                missing_fields += [field]
        
        if len(missing_fields):
            missing_fields = ", ".join(missing_fields)
            raise StrategyException("Missing parameters: {0}".format(missing_fields))
        
    def initialise(self):
        self.check_fields()
        self.indicator = self.measure(self)
        self.forecasts = self.model(self)
        self.positions = self.select_positions(self)
        self.trades = TradeCollection(self)
        if self.filter is not None:
            self.filtered_positions = self.filter(self)
    
    @property
    def trade_timing(self):
        return self.indexer.trade_timing
    
    @property
    def ind_timing(self):
        return self.indexer.ind_timing
    
    def get_indicator_prices(self):
        if self.ind_timing == "C":
            return self.market.close
        if self.ind_timing == "O":
            return self.market.open
        
    @property    
    def lagged_indicator(self):
        return self.indexer.indicator(self.indicator)
    
    @property
    def lagged_positions(self):
        return self.indexer.positions(self.positions)
        
    @property
    def market_returns(self):
        return self.market.returns(self.indexer)
    
    @property
    def returns(self):
        return self.lagged_positions.applied_to(self.market_returns)
    
    @property
    def long_returns(self):
        positions = self.lagged_positions.long_only()
        return positions.applied_to(self.market_returns)
    
    @property
    def short_returns(self):
        positions = self.lagged_positions.short_only()
        return positions.applied_to(self.market_returns)

    @property
    def filtered_lagged_positions(self):
        return self.indexer.positions(self.filtered_positions)

    def plot_returns(self, long_only = False, short_only = False, color = "blue", **kwargs):
        if long_only:
            self.long_returns.plot("sum", color = color, **kwargs)
        elif short_only:
            self.short_returns.plot("sum", color = color, **kwargs)
        else:
            self.returns.plot("sum", color = color, **kwargs)


    def plot_filter_vs_base(self, long_only = True):
        if long_only:
            filt_pos = self.filtered_lagged_positions.long_only()
            base_pos = self.lagged_positions.long_only()
        else:
            filt_pos = self.filtered_lagged_positions
            base_pos = self.lagged_positions
        num = filt_pos.num_concurrent()
        start = num[num > 0].index[0]
        mkt_R = self.market_returns
        filtR = filt_pos.normalised().applied_to(mkt_R)
        baseR = base_pos.normalised().applied_to(mkt_R)
        mkt_R.data = mkt_R.data[start:]
        filtR.data = filtR.data[start:]
        baseR.data = baseR.data[start:]
        markR.plot("mean", color = "black")
        baseR.plot("sum", color = "blue")
        filtR.plot("sum", color = "red")


class StrategyElement(object):
    
    @property
    def ID(self):
        return id(self)
        
    def __call__(self, strategy):
        result = self.get_result(strategy)
        if not self.created(result):
            result = self.execute(strategy)
            result.creator = self.ID
        return result
    
    def execute(self, strategy):
        raise StrategyException("Strategy Element must define 'execute'")
    
    def created(self, result):
        return result is not None and result.creator == self.ID
    
    def get_result(self, strategy):
        raise StrategyException("Strategy Element must define 'get_child'")
    

class MeasureElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.indicator

class ModelElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.forecasts
    
class PositionSelectionElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.positions


class FilterElement(StrategyElement):

    def get_result(self, strategy):
        return strategy.filtered_positions


class StrategyContainerElement(object):
    '''
    StrategyContainerElements represent the data objects used by Strategy.
    Each StrategyContainerElement must have a (DataFrame) field called data.
    '''
    def shift(self, lag):
        lagged = self.copy()
        lagged.data = lagged.data.shift(lag)
        return lagged

    def index(self):
        return self.data.index

    def __getitem__(self, key):
        return self.data[key]

    def copy(self):
        return deepcopy(self)


class Indexer(object):
    
    def __init__(self, trade_timing, ind_timing):
        if trade_timing not in ["OO", "CC", "OC", "CO"]:
            raise StrategyException("Trade timing must be one of: 'OO', 'CC', 'OC', 'CO'.")
        self.trade_timing = trade_timing
        if ind_timing not in ["O", "C"]:
            raise ValueError("Ind_timing must be one of: 'O', 'C'")
        self.ind_timing = ind_timing

    def indicator(self, series):
        if "OC" in self.ind_timing + self.trade_timing:
            lag = 1
        else:
            lag = 2
        return series.shift(lag)
        
    def positions(self, series):
        if "OC" in self.ind_timing + self.trade_timing:
            lag = 1
        else:
            lag = 2
        return series.shift(lag)
    
    def market_returns(self, market):
        timing_map = {"O":"Open", "C":"Close"}
        if self.trade_timing == "OC":
            lag = 0
        else:
            lag = 1
        trade_open = market.get_series(timing_map[self.trade_timing[0]])
        trade_close = market.get_series(timing_map[self.trade_timing[1]])
        return (trade_close / trade_open.shift(lag)) - 1

        
class StrategyException(Exception):
    pass


class EnsembleStrategy(Strategy):
    
    def __init__(self, trade_timing, ind_timing, parameter_set):
        super().__init__(trade_timing, ind_timing)
        self.parameter_set = parameter_set
        self.required_fields += ["forecast_weight"]
        self.strategies = []
        
    def initialise(self):
        self.check_fields()
        self.initialise_substrats()
        self.forecasts = self.forecast_weight(self._forecasts)
        self.positions = self.select_positions(self)
        
    def initialise_substrats(self):
        pass
    
    def build_substrat(self, *args):
        strategy = Strategy(self.trade_timing, self.ind_timing)
        strategy.market = self.market
        strategy.measure = self.measure
        strategy.model = self.model
        strategy.select_positions = self.select_positions
        return strategy
    
    @property
    def _indicators(self):
        return [strategy.indicator for strategy in self.strategies]
    
    @property
    def _forecasts(self):
        return [strategy.forecasts for strategy in self.strategies]
    
    @property
    def _positions(self):
        return [strategy.positions for strategy in self.strategies]
    
    def combined_returns(self, collapse_fun):
        returns = self.returns
        for strategy in self.strategies:
            sub_returns = strategy.returns.collapse_by(collapse_fun)
            returns.append(sub_returns)
        return returns
    
    def plot_returns(self, long_only = False, short_only = False, color = "blue", **kwargs):
        self.plot_substrats(long_only, short_only, color, **kwargs)
        super().plot_returns(long_only, short_only, color = color)
        
    def plot_substrats(self, long_only = False, short_only = False, color = "blue", **kwargs):
        color_map = plt.get_cmap("jet")
        col_index = 80
        col_increment = round(60 / len(self.strategies))
        self.market_returns.plot("mean", color = "black", label = "Market")
        for sub_strat in self.strategies:
            sub_strat.plot_returns(long_only, short_only, color = color_map(col_index), **kwargs)
            col_index += col_increment
    

class MeasureEnsembleStrategy(EnsembleStrategy):
    
    def initialise_substrats(self):
        for parameter in self.parameter_set:
            new_strat = self.build_substrat()
            new_strat.measure.update_param(parameter)
            new_strat.initialise()
            self.strategies += [new_strat]
            
            
class ModelEnsembleStrategy(EnsembleStrategy):
    
    def initialise_substrats(self):
        indicator = self.measure(self)
        for parameter in self.parameter_set:
            new_strat = self.build_substrat()
            new_strat.model.update_param(parameter)
            new_strat.indicator = indicator
            new_strat.initialise()
            self.strategies += [new_strat]
            

class CompoundEnsembleStrategy(EnsembleStrategy):
    
    def initialise_substrats(self):
        model_parameters = self.parameter_set[0]
        measure_parameters = self.parameter_set[1]
        for parameter in model_parameters:
            new_strat = self.build_substrat(measure_parameters)
            new_strat.model.update_param(parameter)
            new_strat.initialise()
            self.strategies += [new_strat]
            
    def build_substrat(self, measure_parameters):
        strategy = MeasureEnsembleStrategy(self.trade_timing, self.ind_timing, measure_parameters)
        strategy.market = self.market
        strategy.measure = self.measure
        strategy.model = self.model
        strategy.select_positions = self.select_positions
        strategy.forecast_weight = self.forecast_weight
        return strategy
    


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
            flags = data.positions.data - data.positions.shift(1).data
            for ticker in self.tickers:
                ticker_flags = flags[ticker]
                T = self.create_trades(ticker_flags, ticker, data.market.close[ticker])
                if len(T) == 0:
                    T = self.create_trades(ticker_flags, ticker, data.market.close[ticker])
                self.trades.extend(T)
        self.plot_series = TradePlotSeriesCollection()
    

    def create_trades(self, flags, ticker, prices):

        entries = flags.index[flags > 0]
        trades = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            i += 1
            if i < len(entries):
                next = entries[i]
            else:
                next = None
            exit = flags[entry:next].index[flags[entry:next] < 0]
            if len(exit) == 0:
                exit = flags.index[-1]
            else:
                exit = exit[0]
            trades.append(Trade(ticker, prices, entry, exit))
        return trades

    def __getitem__(self, key):
        return [trade for trade in self.trades if trade.ticker == key]

    def as_dataframe(self):
        data = [trade.as_tuple() for trade in self.trades]
        return DataFrame(data, columns = self.trades[0].cols)

    def as_dataframe_with(self, values):
        '''
        requires that values is a dataframe with first column ticker
        and remaining columsn values to be inserted.
        Assumes index of values are dates / timestamps.
        '''
        df = self.as_dataframe()
        cols = values.columns[1:]
        for col in cols:
            df[col] = None

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
        sub_trades = filter(condition, list(self.trades))
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


class Trade(object):

    def __init__(self, ticker, prices, entry_date, exit_date):
        self.ticker = ticker
        self.prices = prices
        self.entry = entry_date
        self.exit = exit_date
        self.duration = (exit_date - entry_date).days
        self.normalised = Series((prices[self.entry:self.exit] / prices[self.entry]).values) - 1
        self.cols = ["ticker", "entry", "exit", "price_at_entry", "price_at_exit", "base_return", "duration"]

    def plot_normalised(self):
        self.normalised.plot()

    def as_tuple(self):
        return tuple(getattr(self, name) for name in self.cols)

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
 
           

class TradePlotSeriesCollection(object):

    def __init__(self, color_map = "jet"):
        self.color_map = color_map
        self.collection = DataFrame()

    def plot(self, **kwargs):
        self.collection.plot()

    def append(self, series):
        self.collection[series.name] = series
