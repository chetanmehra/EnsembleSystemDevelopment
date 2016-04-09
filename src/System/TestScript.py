'''
Created on 15 Feb 2015

@author: Mark
'''

from System.Market import Market
from System.Strategy import Strategy, MeasureEnsembleStrategy,\
    ModelEnsembleStrategy, CompoundEnsembleStrategy
from System.Indicator import Crossover
from System.Forecast import BlockForecaster, MeanForecastWeighting
from System.Position import SingleLargestF
import datetime
import matplotlib.pyplot as plt

ASX20 = ["AMP", "ANZ", "BHP", "BXB", "CBA", "CSL", "IAG", "MQG", "NAB", "ORG", "QBE", 
         "RIO", "SCG", "SUN", "TLS", "WBC", "WES", "WFD", "WOW", "WPL"]

def build_market(tickers = ASX20, 
                 start = datetime.datetime(2006, 1, 1), 
                 end = datetime.datetime(2014, 1, 1)):
    
    market = Market(tickers, start, end)
    market.download_data()
    return market
    

def run_various_trade_timings(market):
    strategyCOO = Strategy("OO", "C")
    strategyCOO.market = market
    strategyCOO.measure = Crossover(20, 10)
    strategyCOO.model = BlockForecaster(20)
    strategyCOO.select_positions = SingleLargestF()
    strategyCOO.initialise()
    strategyCOO.market_returns.plot(collapse_fun = "mean", color = "black", label = "BandH")
    strategyCOO.plot_returns(color = "blue", label = "COO")
    
    strategyOCO = Strategy("CO", "O")
    strategyOCO.market = market
    strategyOCO.measure = Crossover(20, 10)
    strategyOCO.model = BlockForecaster(20)
    strategyOCO.select_positions = SingleLargestF()
    strategyOCO.initialise()
    strategyOCO.plot_returns(color = "red", label = "OCO")
    
    strategyOCC = Strategy("CC", "O")
    strategyOCC.market = market
    strategyOCC.measure = Crossover(20, 10)
    strategyOCC.model = BlockForecaster(20)
    strategyOCC.select_positions = SingleLargestF()
    strategyOCC.initialise()
    strategyOCC.plot_returns(color = "orange", label = "OCC")
    
    strategyCOC = Strategy("OC", "C")
    strategyCOC.market = market
    strategyCOC.measure = Crossover(20, 10)
    strategyCOC.model = BlockForecaster(20)
    strategyCOC.select_positions = SingleLargestF()
    strategyCOC.initialise()
    strategyCOC.plot_returns(color = "cyan", label = "COC")
    
    plt.legend()


def run_ensemble_crossover(market, trade_timing = "CC", ind_timing = "O"):
    params = [(25, 15), (20, 10), (15, 10), (12, 8), (8, 5)]
    strategy = MeasureEnsembleStrategy(trade_timing, ind_timing, params)
    strategy.market = market
    strategy.measure = Crossover(1, 1)
    strategy.model = BlockForecaster(20)
    strategy.select_positions = SingleLargestF()
    strategy.forecast_weight = MeanForecastWeighting()
    strategy.initialise()
    return strategy


def run_ensemble_model(market, trade_timing = "CC", ind_timing = "O", 
                       model = BlockForecaster(0), 
                       positions = SingleLargestF()):
    params = [20, 30, 40, 50]
    strategy = ModelEnsembleStrategy(trade_timing, ind_timing, params)
    strategy.market = market
    strategy.measure = Crossover(25, 10)
    strategy.model = model
    strategy.select_positions = positions
    strategy.forecast_weight = MeanForecastWeighting()
    strategy.initialise()
    return strategy

def run_compound_ensemble(market, trade_timing = "CC", ind_timing = "O", 
                          model_params = [20, 30, 40, 50], 
                          measure_params = [(25, 15), (20, 10), (15, 10), (12, 8), (8, 5)], 
                          select_positions = SingleLargestF()):
    strategy = CompoundEnsembleStrategy(trade_timing, ind_timing, [model_params, measure_params])
    strategy.market = market
    strategy.measure = Crossover(0, 0)
    strategy.model = BlockForecaster(0)
    strategy.select_positions = select_positions
    strategy.forecast_weight = MeanForecastWeighting()
    strategy.initialise()
    return strategy



    
    
