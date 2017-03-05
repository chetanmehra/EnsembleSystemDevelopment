'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from copy import copy


class Strategy(object):
    '''
    Strategy holds the components of a trading strategy, including indicators, forecasts etc.
    It manages the creation components as well as the resulting data. 
    '''

    def __init__(self, trade_timing, ind_timing):
        self.indexer = Indexer(trade_timing, ind_timing)
        self.required_fields = ["market", "measure", "model", "select_positions"]
        self.indicator = None
        self.forecasts = None
        self.positions = None
        self.measure = None
        self.model = None
        self.select_positions = None
        
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
    
    def plot_returns(self, long_only = False, short_only = False, color = "blue", **kwargs):
        if long_only:
            self.long_returns.plot("sum", color = color, **kwargs)
        elif short_only:
            self.short_returns.plot("sum", color = color, **kwargs)
        else:
            self.returns.plot("sum", color = color, **kwargs)


class StrategyElement(object):
    
    @property
    def ID(self):
        return id(self)
        
    def __call__(self, strategy):
        child = self.get_child(strategy)
        if not self.is_child(child):
            child = self.execute(strategy)
            child.parent = self.ID
        return child
    
    def execute(self, strategy):
        raise StrategyException("Strategy Element must define 'execute'")
    
    def is_child(self, child):
        if child is None:
            return False
        else:
            return child.parent == self.ID
    
    def get_child(self, strategy):
        raise StrategyException("Strategy Element must define 'get_child'")
    

class MeasureElement(StrategyElement):
    
    def get_child(self, strategy):
        return strategy.indicator

class ModelElement(StrategyElement):
    
    def get_child(self, strategy):
        return strategy.forecasts
    
class PositionSelectionElement(StrategyElement):
    
    def get_child(self, strategy):
        return strategy.positions
    
    
class StrategyContainerElement(object):
    '''
    StrategyContainerElements represent the data objects used by Strategy.
    Each StrategyContainerElement must have a (DataFrame) field called data.
    '''
    def shift(self, lag):
        lagged = copy(self)
        lagged.data = self.data.shift(lag)
        return lagged

    def index(self):
        return self.data.index

    def __getitem__(self, key):
        return self.data[key]


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
    


