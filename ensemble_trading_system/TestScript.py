'''
Created on 15 Feb 2015

@author: Mark
'''
from multiprocessing import Pool
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
# Add dev folder of related package(s)
import sys
import os
sys.path.append(os.path.join("C:\\Users", os.getlogin(), "Source\\Repos\\FinancialDataHandling\\financial_data_handling"))
from store.file_system import Storage

# The following is due to working directory not being set correctly on workstation:
#if os.getlogin() == "Mark":
os.chdir(os.path.join("C:\\Users", os.getlogin(), "Source\\Repos\\EnsembleSystemDevelopment\\ensemble_trading_system"))

# Local imports
from data_types.market import Market
from data_types.trades import TradeCollection

from system.core import Strategy, Portfolio
from system.core import VolatilityMultiplier, SizingStrategy

from measures.moving_averages import EMA, KAMA, LinearTrend
from measures.volatility import StdDevEMA
from measures.breakout import TrailingHighLow
from measures.valuations import ValueRatio, ValueRank
from signals.level_signals import Crossover, Breakout
from signals.carter_forecasters import PriceCrossover, EWMAC, CarterForecastFamily
from rules.signal_rules import PositionFromDiscreteSignal
from rules.forecast_rules import CarterPositions

from trade_modifiers.exit_conditions import StopLoss, TrailingStop, ReturnTriggeredTrailingStop
from trade_modifiers.filters import HighPassFilter
from system.core import PositionCostThreshold, PositionMaxSize, PositionMinSize
from system.analysis import summary_report, ParameterFuzzer


# TODO Strategy should keep full list of trades after filter, to allow easy reapplication of filters without having to
# run full strategy again.

# TODO Strategy idea
# Conditions - Entry
#   - Market index EMA(50-200) must be long
#   - Individual ticker EMA(50-120) must be long
#   - Ticker value ratio (Adjusted) > 2
#   - Value rank filter at 10
# Conditions - Exit
#   - Individual ticker EMA(50-120) goes short
#   - 15% Stop loss
#   - Apply 20% trailing stop when returns exceed 30%
#   - Apply 10% trailing stop when returns exceed 50%


pd.set_option('display.width', 120)

store = Storage("NYSE")

def signalStratSetup(trade_timing = "O", ind_timing = "C", params = (120, 50), store = store):
    strategy = Strategy(trade_timing, ind_timing)
    strategy.market = Market(store)
    strategy.signal_generator = Crossover(slow = EMA(params[0]), fast = EMA(params[1]))
    strategy.position_rules = PositionFromDiscreteSignal(Up = 1)
    return strategy

def createEwmacStrategy(ema_params = (120, 50), store = store):
    strategy = Strategy("O", "C")
    strategy.market = Market(store)
    strategy.signal_generator = EWMAC(EMA(max(ema_params)), EMA(min(ema_params)), StdDevEMA(36))
    strategy.position_rules = CarterPositions(StdDevEMA(36), 0.25, long_only = True)
    return strategy

def createBreakoutStrategy(window = 20, store = store):
    strategy = Strategy("O", "C")
    strategy.market = Market(store)
    strategy.signal_generator = Breakout(TrailingHighLow(window))
    strategy.position_rules = PositionFromDiscreteSignal(Up = 1)
    return strategy

def createValueRatioStrategy(ema_pd = 90, valuations = ["Adjusted", "Base", "Min"], store = store):
    strategy = Strategy("O", "C")
    strategy.market = Market(store)
    signal_generators = []
    for valuation in valuations:
        value_ratio = ValueRatio("EPV", valuation)
        value_ratios = value_ratio(strategy)
        signal_generators.append(PriceCrossover(value_ratios, EMA(ema_pd), StdDevEMA(36)))
    strategy.signal_generator = CarterForecastFamily(*signal_generators)
    strategy.position_rules = CarterPositions(StdDevEMA(36), 0.25, long_only = True)
    return strategy

print("Preparing strat...")
strat = signalStratSetup('O', 'C', params = (150, 75))
##strat = createBreakoutStrategy(window = 60)

# adjusted = ValueRatio('EPV', 'Adjusted')(strat)
# base = ValueRatio('EPV', 'Base')(strat)
# cyclic = ValueRatio('EPV', 'Cyclic')(strat)
# strat.filters.append(HighPassFilter(adjusted, -0.5))
# strat.filters.append(HighPassFilter(cyclic, 0.0))

print("Running base strat...")
strat.run()
print("Generated", strat.trades.count, "trades.")

# print("Applying stops...")
# strat.apply_exit_condition(TrailingStop(0.15))
# strat.apply_exit_condition(StopLoss(0.15))
# strat.apply_exit_condition(ReturnTriggeredTrailingStop(0.2, 0.3))
# strat.apply_exit_condition(ReturnTriggeredTrailingStop(0.1, 0.5))

#with open(r'D:\Investing\Workspace\signal_strat.pkl', 'rb') as file:
#    strat = pickle.load(file)

#print("Creating ewmac strat...")
#strat = createEwmacStrategy(store)
#print("Running ewmac strat...")
#strat.run()
#strat.positions = strat.positions.discretise(min_size = 0.7, max_size = 2.0, step = 0.5)

#print("Loading strat...")
#with open(r'D:\Investing\Workspace\test_strat.pkl', 'rb') as file:
#    strat = pickle.load(file)

# print("Preparing portfolio...")
# port = Portfolio(strat, 15000)
# port.sizing_strategy = SizingStrategy(diversifier = 0.5)
# port.position_checks.append(PositionCostThreshold(0.02))
# print("Running portfolio...")
# port.run()
# print("Done...")


# from system.analysis import ParameterFuzzer
# fuzzer = ParameterFuzzer(strategy, base_parameters = (150, 75), processes = 2)
# fuzzer.fuzzed_pars = [(200, 160), (200, 100), (200, 40), (150, 120), (150, 75), (150, 30), (100, 80), (100, 50), (100, 20)]
# fuzzer.summarise()
# fig, axarr = plt.subplots(1, 3)
# fuzzer.plot_metric('Number of trades', ax = axarr[0])
# fuzzer.plot_metric('Percent winners', ax = axarr[1])
# fuzzer.plot_metric('Ratio average win to loss', ax = axarr[2])
# plt.tight_layout()

# print("Applying weights...")
# from trade_modifiers.weighting import TradeWeighting
# volatilities = StdDevEMA(40)(strat.indicator_prices)
# strat.trades = strat.trades.apply(TradeWeighting(0.3 / volatilities))


print("Ready...")

