'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from pandas import DateOffset, Panel, DataFrame, Series


class Strategy(object):
    '''
    Strategy holds the components of a trading strategy, including indicators, forecasts etc.
    It manages the creation components as well as the resulting data. 
    '''

    def __init__(self, trade_timing, ind_timing):
        self.indexer = Indexer(trade_timing, ind_timing)
        self.required_fields = ["market", "measure", "model", "select_positions"]
        self.name = None
        # Container elements
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.filtered_positions = None
        # Strategy creation elements
        self.measure = None
        self.model = None
        self.select_positions = None
        self.filter = None

    def __str__(self):
        if self.name is not None:
            first_line = self.name
        else:
            first_line = ''
        second_line = 'Measure:\t{}\n'.format(self.measure.name)
        third_line = 'Model:\t{}\n'.format(self.model.name)
        fourth_line = 'Position:\t{}\n'.format(self.select_positions.name)
        if self.filter is not None:
            fifth_line = 'Filter:\t{}\n'.format(self.filter.name)
        else:
            fifth_line = 'No filter specified.\n'
        return first_line + second_line + third_line + fourth_line + fifth_line
        
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
        self.name = self.measure.name
        self.indicator = self.measure(self)
        self.forecasts = self.model(self)
        self.positions = self.select_positions(self)
        if self.filter is not None:
            self.apply_filter(self.filter)
            
            
    def refresh(self):
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.initialise()


    def apply_filter(self, filter):
        self.positions = filter(self)
            
                
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

    
    @property
    def trades(self):
        return self.positions.trades
            
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

    def filter_summary(self, filter_values, boundaries):
        return self.positions.filter_summary(filter_values, boundaries)

    def filter_comparison(self, filter_values, filter1_type, filter2_type, boundaries1, boundaries2):
        return self.positions.filter_comparison(filter_values, filter1_type, filter2_type, boundaries1, boundaries2)

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
        if self.filter is not None:
            self.filter.plot(ticker, start, end, ax)
        self.indicator.plot_measures(ticker, start, end, ax)
        lo_entry = self.market.low[ticker][entries] * 0.95
        hi_exit = self.market.high[ticker][exits] * 1.05
        plt.scatter(entries, lo_entry, marker = '^', color = 'green')
        plt.scatter(exits, hi_exit, marker = 'v', color = 'red')
        plt.title(ticker)


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

    def __call__(self, strategy):
        result = super(PositionSelectionElement, self).__call__(strategy)
        result.create_trades(strategy.get_entry_prices(), strategy.get_exit_prices())
        return result


class StrategyContainerElement(object):
    '''
    StrategyContainerElements represent the data objects used by Strategy.
    Each StrategyContainerElement is assumed to have a (DataFrame) field called data.
    '''
    def shift(self, lag):
        lagged = self.copy()
        lagged.data = lagged.data.shift(lag)
        return lagged

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
        return (trade_close / trade_open.shift(lag)) - 1

        
class StrategyException(Exception):
    pass


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

    @property
    def names(self):
        return [strategy.name for strategy in self.strategies]
    
    def combined_returns(self, collapse_fun):
        returns = self.returns
        for strategy in self.strategies:
            sub_returns = strategy.returns.collapse_by(collapse_fun)
            returns.append(sub_returns)
        return returns

    def filter_summary(self, filter_values, boundaries):

        mu = {}
        sd = {}
        N = {}

        for strategy in self.strategies:
            print("Calculating strat: " + strategy.name)
            summary = strategy.filter_summary(filter_values, boundaries)
            mu[strategy.name] = summary["mean"]
            sd[strategy.name] = summary["std"]
            N[strategy.name] = summary["count"]
            

        return {"mean" : Panel(mu), "std" : Panel(sd), "count" : Panel(N)}

    
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
    

           
