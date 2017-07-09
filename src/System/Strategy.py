'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from pandas import DateOffset, Panel, DataFrame, Series
from System.Trade import Trade, TradeCollection, createTrades



class Strategy(object):
    '''
    Strategy defines the base interface for Strategy objects
    '''
    def __init__(self, trade_timing, ind_timing):
        self.indexer = Indexer(trade_timing, ind_timing)
        self.required_fields = ["market", "signal_generator", "position_rules"]
        self.name = None
        self.signal = None
        self.positions = None
        self.signal_generator = None
        self.position_rules = None
        self.filters = []

    def __str__(self):
        if self.name is not None:
            first_line = '{}\n'.format(self.name)
        else:
            first_line = ''
        second_line = 'Signal:\t{}\n'.format(self.signal_generator.name)
        third_line = 'Position Rule:\t{}\n'.format(self.position_rules.name)
        if len(self.filters) > 0:
            fourth_line = 'Filter:\t{}\n'.format('\n'.join([f.name for f in self.filters]))
        else:
            fourth_line = 'No filter specified.\n'
        return first_line + second_line + third_line + fourth_line

    def run(self):
        self.generate_signals()
        self.apply_rules()
        self.apply_filters()
        
    def rerun(self):
        self.reset()
        self.run()

    def reset(self):
        self.signal = None
        self.positions = None
        self.trades = None


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
            
    def apply_rules(self):
        self.positions = self.position_rules(self)
        self.trades = createTrades(self.positions.data, self)

    def generate_signals(self):
        self.signal = self.signal_generator(self)

    def apply_filters(self):
        if len(self.filters) == 0:
            return
        for filter in self.filters:
            self.trades = filter(self)
        self.positions.updateFromTrades(self.trades)

    def applyFilter(self, filter):
        self.filters += [filter]
        self.trades = filter(self)
        self.positions.updateFromTrades(self.trades)
    
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

    def get_trade_prices(self):
        if self.trade_timing[0] == "O":
            return self.market.open
        else:
            return self.market.close

    def getEmptyDataFrame(self, fill_data = None):
        return self.market.getEmptyDataFrame(fill_data)

    def buyAndHoldTrades(self):
        signal_data = self.getEmptyDataFrame()
        signal_data[:] = 1
        return createTrades(signal_data, self)

    @property
    def marketReturns(self):
        return self.market.returns(self.indexer)
    
    @property
    def returns(self):
        return self.positions.appliedTo(self.marketReturns)
    
    @property
    def longReturns(self):
        positions = self.positions.long_only()
        return positions(self)
    
    @property
    def shortReturns(self):
        positions = self.positions.short_only()
        return positions(self)

    def plot_measures(self, ticker, start, end, ax):
        raise NotImplementedError("Strategy must override plot_measures")

    def plotReturns(self, long_only = False, short_only = False, color = "blue", **kwargs):

        if long_only:
            returns = self.longReturns
        elif short_only:
            returns = self.shortReturns
        else:
            returns = self.returns
        start = self.positions.start
        returns.plot(start = start, color = color, **kwargs)
        self.marketReturns.plot(start = start, color = "black", label = "Market")
        plt.legend(loc = "upper left")


    def plotTrade(self, key):
        trades = self.trades[key]
        if isinstance(trades, list):
            if len(trades) == 0:
                print('No trades for {}'.format(key))
                return
            entries, exits = zip(*[(T.entry, T.exit) for T in trades])
            entries = list(entries)
            exits = list(exits)
            ticker = key
        else:
            entries = [trades.entry]
            exits = [trades.exit]
            ticker = trades.ticker
        start = min(entries) - DateOffset(10)
        end = max(exits) + DateOffset(10)
        fig, ax = self.market.candlestick(ticker, start, end)
        self.indicator.plot_measures(ticker, start, end, ax)
        for filter in self.filters:
            filter.plot(ticker, start, end, ax)
        lo_entry = self.market.low[ticker][entries] * 0.95
        hi_exit = self.market.high[ticker][exits] * 1.05
        plt.scatter(entries, lo_entry, marker = '^', color = 'green')
        plt.scatter(exits, hi_exit, marker = 'v', color = 'red')
        plt.title(ticker)


class StrategyElement:
    
    @property
    def ID(self): 
        return id(self)
        
    def __call__(self, strategy):
        result = self.get_result(strategy)
        if not self.created(result):
            result = self.execute(strategy)
            result.creator = self.ID
            result.indexer = strategy.indexer
            result.calculation_timing = self.calculation_timing
            result.lag = self.starting_lag()
        return result

    
    def execute(self, strategy):
        raise NotImplementedError("Strategy Element must define 'execute'")
    
    def created(self, result):
        return result is not None and result.creator == self.ID
    
    def get_result(self, strategy):
        raise NotImplementedError("Strategy Element must define 'get_child'")

    def starting_lag(self):
        return 0
    

class SignalElement(StrategyElement):

    def get_result(self, strategy):
        return strategy.signal

    @property
    def calculation_timing(self):
        return ["decision"]

    
class PositionRuleElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.positions

    @property
    def calculation_timing(self):
        return ["entry"]


class DataElement:
    '''
    DataElements represent the data objects used by Strategy.
    Each DataElement is assumed to have a (DataFrame) field called data.
    DataElements also control the appropriate lagging of data to align with other calculation periods.
    When lagged the DataElement returns a copy of itself, and the current applied lag.
    Calculation timing represents the timing for when the data is calculated and may be a single value, 
    e.g. ["decision", "entry", "exit", "open", "close"], or a twin value which represents the calculations 
    occur over some span of time, e.g. for returns which have timing ["entry", "exit"]
    '''
    def __init__(self):
        self.lag = 0
        self.calculation_timing = None

    def shift(self, lag):
        lagged = self.copy()
        lagged.data = self.data.copy() # Note: deepcopy doesn't seem to copy the dataframe properly; hence this line.
        lagged.data = lagged.data.shift(lag)
        lagged.lag += lag
        return lagged

    def at(self, timing):
        '''
        'at' returns a lagged version of the data appropriate for the timing.
        Timing may be one of: ["decision", "entry", "exit", "open", "close"]
        '''
        lag = self.get_lag(timing)
        return self.shift(lag)

    def alignWith(self, other):
        '''
        'alignWith' returns a lagged version of the data so that it is aligned with the 
        timing of the supplied other data object. This ensures that this data will be available at the 
        time creation of the other data object would have started.
        The logic for the overall lag calculated in alignWith is as follows:
        - total lag = alignment lag + other lag
            - alignment lag = lag between this data calc end to the other data calc start
            - other lag = the lag already applied to the other data.
        '''
        alignment_lag = self.get_lag(other.calculation_timing[0])
        total_lag = alignment_lag + other.lag
        return self.shift(total_lag)

    def get_lag(self, target_timing):
        '''
        Calculate the lag between the completion of calculating this data object and the requested timing.
        '''
        return self.indexer.getLag(self.calculation_timing[-1], target_timing)

    @property
    def index(self):
        return self.data.index

    @property
    def tickers(self):
        return self.data.columns

    def __getitem__(self, key):
        return self.data[key]

    def copy(self):
        return deepcopy(self)


class Indexer(object):
    
    def __init__(self, trade_timing, ind_timing):
        if trade_timing not in ["OO", "CC"]:
            raise ValueError("Trade timing must be one of: 'OO', 'CC'.")
        self.trade_timing = trade_timing
        if ind_timing not in ["O", "C"]:
            raise ValueError("Ind_timing must be one of: 'O', 'C'")
        self.ind_timing = ind_timing
        self.timing_map = {"decision" : self.ind_timing, 
                           "entry" : self.trade_timing[0], 
                           "exit" : self.trade_timing[-1], 
                           "open" : "O", 
                           "close" : "C"}


    def getLag(self, start, end):
        start = self.check_timing(start)
        end = self.check_timing(end)
        if "OC" in start + end:
            lag = 0
        else:
            lag = 1
        return lag

    def check_timing(self, timing):
        timing = timing.lower()
        if timing not in self.timing_map.keys():
            raise ValueError("Timing must be one of: " + ",".join(self.timing_map.keys()))
        return self.timing_map[timing]
    
    def marketReturns(self, market):
        timing_map = {"O":"open", "C":"close"}
        if self.trade_timing == "OC":
            lag = 0
        else:
            lag = 1
        trade_open = getattr(market, timing_map[self.trade_timing[0]])
        trade_close = getattr(market, timing_map[self.trade_timing[1]])
        return (trade_open / trade_close.shift(lag)) - 1

        
class EnsembleStrategy(Strategy):
    
    def __init__(self, trade_timing, ind_timing, parameter_set):
        super(EnsembleStrategy, self).__init__(trade_timing, ind_timing)
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
        strategy = ModelStrategy(self.trade_timing, self.ind_timing)
        strategy.market = self.market
        strategy.measure = self.measure
        strategy.model = self.model
        strategy.select_positions = self.select_positions
        return strategy
    
    @property
    def _indicators(self):
        return [strategy.level_indicator for strategy in self.strategies]
    
    @property
    def _forecasts(self):
        return [strategy.forecasts for strategy in self.strategies]
    
    @property
    def _positions(self):
        return [strategy.positions for strategy in self.strategies]

    @property
    def names(self):
        return [strategy.name for strategy in self.strategies]
    
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
        self.market_returns.plot(color = "black", label = "Market")
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
    

           
