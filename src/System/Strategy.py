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
        self.required_fields = ["market"]
        self.name = None

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

    def run(self):
        self.check_fields()
        self.initialise()
        
    def rerun(self):
        self.reset()
        self.run()
        
    def initialise(self):
        raise NotImplementedError("Strategy must override initialise")
  
    def reset(self):
        raise NotImplementedError("Strategy must override reset")

    def apply_filter(self, filter):
        self.filters += [filter]
        self.trades = filter(self)

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

    def get_entry_prices(self):
        if self.trade_timing[0] == "O":
            return self.market.open
        else:
            return self.market.close

    def get_exit_prices(self):
        if self.trade_timing[1] == "O":
            return self.market.open
        else:
            return self.market.close

    def get_empty_dataframe(self):
        return self.market.get_empty_dataframe()

    def buy_and_hold_trades(self):
        signal_data = self.get_empty_dataframe()
        signal_data[:] = 1
        return createTrades(signal_data, self)
            
    @property    
    def lagged_indicator(self):
        return self.indexer.indicator(self.indicator)

    @property
    def market_returns(self):
        return self.market.returns(self.indexer)
    
    @property
    def returns(self):
        return self.positions.applied_to(self.market_returns)
    
    @property
    def long_returns(self):
        positions = self.positions.long_only()
        return positions.applied_to(self.market_returns)
    
    @property
    def short_returns(self):
        positions = self.positions.short_only()
        return positions.applied_to(self.market_returns)

    def plot_measures(self, ticker, start, end, ax):
        raise NotImplementedError("Strategy must override plot_measures")

    def plot_returns(self, long_only = False, short_only = False, color = "blue", **kwargs):

        if long_only:
            returns = self.long_returns
        elif short_only:
            returns = self.short_returns
        else:
            returns = self.returns
        start = self.positions.start
        returns.plot(start = start, color = color, **kwargs)
        self.market_returns.plot(start = start, color = "black", label = "Market")
        plt.legend(loc = "upper left")


    def plot_trade(self, key):
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


class SignalStrategy(Strategy):
    '''
    A SignalStrategy is the more fundamental strategy type involving entry and exit signals derived from
    one or more indicators and signal generators.
    '''
    def __init__(self, trade_timing, ind_timing):
        super(SignalStrategy, self).__init__(trade_timing, ind_timing)
        self.required_fields += ["signal"]
        self.name = 'Signal Strategy'
        # Container elements
        self.indicator = None
        self.trades = None
        # Strategy creation elements
        self.signal = None
        self.filters = []

    def initialise(self):
        self.indicator = self.signal(self)
        self.trades = createTrades(self.lagged_indicator.data.astype(float), self)
        for filter in self.filters:
            self.trades = filter(self)

    def reset(self):
        self.indicator = None
        self.trades = None


class AltStrategy(Strategy):
    '''
    Provides the interface for collating and testing trading hypotheses
    '''
    def __init__(self, trade_timing, ind_timing):
        super().__init__(trade_timing, ind_timing)
        self.name = "Strategy"
        self.signal = None
        self.positions = None
        self.filters = []

    def run(self):
        self.generateSignals()
        self.applyRules()
        self.applyFilters()

    def generateSignals(self):
        self.signal = self.signal_generator(self)
        
    def applyRules(self):
        self.positions = self.position_rules(self)
        self.trades = createTrades(self.positions.data, self)

    def applyFilters(self):
        for filter in self.filters:
            self.trades = filter(self)
        self.positions.updateFromTrades(self.trades)


class ModelStrategy(Strategy):
    '''
    ModelStrategy holds the components of a trading strategy based on forecasting, including indicators, forecasts etc.
    It manages the creation components as well as the resulting data. 
    '''

    def __init__(self, trade_timing, ind_timing):
        super().__init__(trade_timing, ind_timing)
        self.required_fields += ["measure", "model", "select_positions"]
        self.name = 'Model Strategy'
        # Container elements
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.trades = None
        # Strategy creation elements
        self.measure = None
        self.model = None
        self.select_positions = None
        self.filters = []

    def __str__(self):
        if self.name is not None:
            first_line = '{}\n'.format(self.name)
        else:
            first_line = ''
        second_line = 'Measure:\t{}\n'.format(self.measure.name)
        third_line = 'Model:\t{}\n'.format(self.model.name)
        fourth_line = 'Position:\t{}\n'.format(self.select_positions.name)
        if len(self.filters) > 0:
            fifth_line = 'Filter:\t{}\n'.format('\n'.join([f.name for f in self.filters]))
        else:
            fifth_line = 'No filter specified.\n'
        return first_line + second_line + third_line + fourth_line + fifth_line
        
    # HACK ModelStrategy filters don't feed back into positions
    def initialise(self):
        self.indicator = self.measure(self)
        self.forecasts = self.model(self)
        self.positions = self.select_positions(self)
        self.trades = createTrades(self.positions.data, self)
        for filter in self.filters:
            self.trades = filter(self)
            
    def reset(self):
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.trades = None

            

class StrategyElement(object):
    
    @property
    def ID(self): 
        return id(self)
        
    def __call__(self, strategy):
        result = self.get_result(strategy)
        if not self.created(result):
            result = self.execute(strategy)
            result.creator = self.ID
            result.indexer = strategy.indexer
            result.calculation_timing = self.get_calculation_timing(strategy)
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

    def get_calculation_timing(self, strategy):
        return strategy.indexer.ind_timing

class ModelElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.forecasts

    def get_calculation_timing(self, strategy):
        return strategy.indexer.ind_timing
    
class PositionSelectionElement(StrategyElement):
    
    def get_result(self, strategy):
        return strategy.positions

    def get_calculation_timing(self, strategy):
        return strategy.indexer.trade_timing[0]


class ReturnsElement(StrategyElement):

    def get_result(self, strategy):
        return None

    def get_calculation_timing(self, strategy):
        return strategy.indexer.trade_timing

class StrategyContainerElement(object):
    '''
    StrategyContainerElements represent the data objects used by Strategy.
    Each StrategyContainerElement is assumed to have a (DataFrame) field called data.
    '''
    def shift(self, lag):
        lagged = self.copy()
        lagged.data = lagged.data.shift(lag)
        return lagged

    def at(self, timing, align_with = None):
        '''
        'at' returns a lagged version of the data appropriate for the timing, and alignment.
        The indexer is relied on to calculate the appropriate lag values.
        timing and align_with may be one of: "decision", "entry", "exit"
        align_with may also be "returns" which is effectively the same as "exit" but provided for clarity.
        '''
        lag = self.get_lag(timing)
        if align_with is not None:
            lag += self.get_lag(align_with)
        return self.shift(lag)

    def get_lag(self, timing):
        timing = timing.lower()
        if timing == "decision":
            timing_lag = 0
        elif timing == "entry":
            timing_lag = 1 #* ("OC" in self.indexer.ind_timing + self.indexer.trade_timing)
        elif timing in ["exit", "returns"]:
            timing_lag = 1 #+ ("OC" in self.indexer.ind_timing + self.indexer.trade_timing)
        return timing_lag

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
        timing_map = {"O":"open", "C":"close"}
        if self.trade_timing == "OC":
            lag = 0
        else:
            lag = 1
        trade_open = getattr(market, timing_map[self.trade_timing[0]])
        trade_close = getattr(market, timing_map[self.trade_timing[1]])
        return (trade_open / trade_close.shift(lag)) - 1

        
class StrategyException(Exception):
    pass


class EnsembleStrategy(ModelStrategy):
    
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
    

           
