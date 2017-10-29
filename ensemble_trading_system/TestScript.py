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
from data_types.filter_data import StackedFilterValues, WideFilterValues

from system.core import Strategy, Portfolio
from level_signals import Crossover
from measures.moving_averages import EMA, KAMA
from measures.volatility import StdDevEMA
from carter_forecasters import PriceCrossover, EWMAC, CarterForecastFamily
from rules.signal_rules import PositionFromDiscreteSignal
from rules.forecast_rules import CarterPositions

from trade_modifiers.exit_conditions import StopLoss, TrailingStop, ReturnTriggeredTrailingStop
from trade_modifiers.filters import HighPassFilter
from system.core import PositionCostThreshold, PositionMaxSize, PositionMinSize
from system.analysis import summary_report, ParameterFuzzer


# TODO Compare full valuation calculations (e.g. from statements) with simplified valuations (from CMC summary)
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

# TODO Refactoring
#   - Inherit strategy data element from dataframe


pd.set_option('display.width', 120)

ASX20 = ["AMP", "ANZ", "BHP", "BXB", "CBA", "CSL", "IAG", "MQG", "NAB", "ORG", "QBE", 
         "RIO", "SCG", "SUN", "TLS", "WBC", "WES", "WFD", "WOW", "WPL"]

triplet = ["MLD", "CCP", "BHP"]


valued_tickers = [u'ONT', u'TGP', u'TIX', u'TOF', u'3PL', u'ABP', u'ALR', u'ACR',
                  u'ADA', u'AAU', u'ASW', u'AGL', u'AGJ', u'APW', u'AGI', u'AIZ',
                  u'AQG', u'LEP', u'ALU', u'AMA', u'AMB', u'AMH', u'AMC', u'AGF',
                  u'ANN', u'ANO', u'APE', u'APA', u'APN', u'APD', u'ARB', u'AAD',
                  u'ARF', u'ARG', u'ALL', u'AIK', u'ASZ', u'ASH', u'AJJ', u'AUF',
                  u'AJA', u'ATR', u'ASX', u'AUB', u'AIA', u'AZJ', u'AUP', u'AST',
                  u'AGD', u'AAC', u'AAP', u'AEF', u'AFI', u'AKY', u'API', u'ARW',
                  u'AUI', u'AVG', u'AHG', u'AOG', u'AVJ', u'AXI', u'AZV', u'BLX',
                  u'BCN', u'BFG', u'BHP', u'BGL', u'BXN', u'BIS', u'BKI', u'BGG',
                  u'BWF', u'BWR', u'BLA', u'BSL', u'BLD', u'BXB', u'BRG', u'BKW',
                  u'BYL', u'BBL', u'BPA', u'BSA', u'BTT', u'BUG', u'BAP', u'BWP',
                  u'BPG', u'CAB', u'CDM', u'CTX', u'CZZ', u'CAJ', u'CAA', u'CDP',
                  u'CIN', u'CAR', u'CWP', u'CLT', u'CAF', u'CNI', u'CYA', u'CHC',
                  u'CQR', u'CII', u'CSS', u'CVW', u'CIW', u'CLV', u'CMI', u'CCL',
                  u'CDA', u'CLH', u'CKF', u'CMP', u'CPU', u'CTD', u'COO', u'CUP',
                  u'CCP', u'CMW', u'CWN', u'CTE', u'CSV', u'CSL', u'CSR', u'CLX',
                  u'CUE', u'CYC', u'DTL', u'DCG', u'DSB', u'DGH', u'DVN', u'DXS',
                  u'DUI', u'DJW', u'DMP', u'DRM', u'DOW', u'DRA', u'DWS', u'ELX',
                  u'EMB', u'EGO', u'EPD', u'EOL', u'EWC', u'ETE', u'EGL', u'EQT',
                  u'EPW', u'EBG', u'EGH', u'FAN', u'FRM', u'FFI', u'FID', u'FRI',
                  u'FPH', u'FSI', u'FWD', u'FBU', u'FXL', u'FLT', u'FET', u'FLK',
                  u'FSF', u'FMG', u'FNP', u'FSA', u'FGX', u'GUD', u'GEM', u'GAP',
                  u'GJT', u'GZL', u'GBT', u'GHC', u'GLE', u'GCS', u'GLH', u'GFL',
                  u'GLB', u'GMG', u'GOW', u'GPT', u'GNG', u'GNC', u'GXL', u'GOZ',
                  u'HSN', u'HVN', u'HGG', u'HNG', u'HIT', u'HOM', u'HZN', u'HPI',
                  u'HHV', u'HHL', u'ICS', u'IMF', u'IPL', u'IGO', u'IFN', u'IFM',
                  u'INA', u'ITQ', u'IRI', u'IOF', u'IVC', u'IFL', u'IPE', u'IRE',
                  u'IBC', u'ISD', u'ITD', u'JYC', u'JIN', u'KSC', u'KAM', u'KBC',
                  u'KRM', u'KME', u'KKT', u'KOV', u'LMW', u'LBL', u'LGD', u'LLC',
                  u'LHC', u'LIC', u'LAU', u'LCM', u'LCE', u'MLD', u'MQA', u'MRN',
                  u'MFG', u'MFF', u'MTR', u'MCE', u'MXI', u'MSP', u'MYX', u'MMS',
                  u'MCP', u'MVP', u'MLB', u'MLX', u'MWR', u'MLT', u'MRC', u'MIR',
                  u'MGR', u'MNF', u'MBE', u'MND', u'MVF', u'MNY', u'MOC', u'MYR',
                  u'NMS', u'NTC', u'NCM', u'NHH', u'NCK', u'NST', u'NUF', u'OCL',
                  u'OGC', u'OCP', u'OSH', u'OEC', u'ORL', u'OFX', u'PEA', u'PFM',
                  u'PGC', u'PFL', u'PAY', u'PPC', u'PTL', u'PPT', u'PRU', u'PXS',
                  u'PHI', u'PTM', u'PMC', u'PRT', u'PME', u'PPG', u'PRO', u'PSZ',
                  u'PTB', u'PHA', u'QAN', u'QTM', u'RMS', u'RHC', u'RND', u'RCG',
                  u'REA', u'RKN', u'RFT', u'RDH', u'REH', u'RCT', u'REX', u'RRL',
                  u'RHT', u'RDG', u'RFG', u'REF', u'RIS', u'RIC', u'RIO', u'RWH',
                  u'RFF', u'RXP', u'SAI', u'SFR', u'SAR', u'SND', u'SFC', u'SDI',
                  u'SLK', u'SEK', u'SHV', u'SEN', u'SRV', u'SSM', u'SWL', u'SGF',
                  u'SHU', u'SHJ', u'SIP', u'SIV', u'SRX', u'SIT', u'SKT', u'SKC',
                  u'SGH', u'SMX', u'SOM', u'SHL', u'SKI', u'SPK', u'SPO', u'SDF',
                  u'SST', u'SGP', u'SDG', u'SUL', u'SNL', u'TAH', u'TWD', u'TMM',
                  u'TGR', u'TTS', u'TCN', u'TNE', u'TLS', u'TGG', u'TFC', u'TRS',
                  u'SGR', u'TSM', u'TGA', u'TOP', u'TOX', u'TPM', u'TME', u'TTI',
                  u'TCO', u'TWE', u'TBR', u'UOS', u'UPG', u'UNV', u'VII', u'VLW',
                  u'VRT', u'VTG', u'VSC', u'VMT', u'VOC', u'WAA', u'WAM', u'WAX',
                  u'SOL', u'WAT', u'WEB', u'WBA', u'WLL', u'WES', u'WSA', u'WHF',
                  u'WIG', u'WPL', u'WOW', u'WRR', u'XRF', u'XTE', u'ZGL']

dodgy_tickers = [u'MWR', u'FGX', u'NMS', u'ARW', u'SOM', u'GJT', 
                 u'ICS', u'XTE', u'EGO', u'BXN', u'DRA', u'VMT', 
                 u'PGC']

good_tickers = list(set(valued_tickers) - set(dodgy_tickers))

def getMarket(store, source = None, excluded_tickers = None):
    if source is None:
        source = store.exchange
    return Market(store.get_instruments(source, excluded_tickers))


def getValues(store, date = None):
    valuations = store.get_valuations(date)
    return StackedFilterValues(valuations.summary, "Valuations")

def getValueRatios(store, value_type, strat):
    valuation = getValues(store).as_wide_values(value_type)
    ratios = valuation.value_ratio(strat.get_trade_prices().data)
    return ratios

def getValueMetrics(store, date = None):
    metrics = store.get_valuemetrics(date)
    return StackedFilterValues(metrics.summary, "ValuationMetrics")


def signalStratSetup(trade_timing = "C", ind_timing = "O", params = (120, 50), exchange = "NYSE"):
    store = Storage(exchange)
    strategy = Strategy(trade_timing, ind_timing)
    strategy.market = getMarket(store)
    strategy.signal_generator = Crossover(slow = EMA(params[0]), fast = EMA(params[1]))
    strategy.position_rules = PositionFromDiscreteSignal(Up = 1)
    return strategy

def createEwmacStrategy(store, ema_params = (120, 50)):
    strategy = Strategy("O", "C")
    strategy.market = getMarket(store)
    strategy.signal_generator = EWMAC(EMA(max(ema_params)), EMA(min(ema_params)), StdDevEMA(36))
    strategy.position_rules = CarterPositions(StdDevEMA(36), 0.25, long_only = True)
    return strategy

def createValueRatioStrategy(store, ema_pd = 90, valuations = ["Adjusted", "Base", "Min"]):
    strategy = Strategy("O", "C")
    strategy.market = getMarket(store)
    signal_generators = []
    for valuation in valuations:
        value_ratios = getValueRatios(store, valuation, strategy)
        signal_generators.append(PriceCrossover(value_ratios, EMA(ema_pd), StdDevEMA(36)))
    strategy.signal_generator = CarterForecastFamily(*signal_generators)
    strategy.position_rules = CarterPositions(StdDevEMA(36), 0.25, long_only = True)
    return strategy



store = Storage("NYSE")

strat = signalStratSetup('O', 'C')
adjusted = getValueRatios(store, 'Adjusted', strat)
base = getValueRatios(store, 'Base', strat)
cyclic = getValueRatios(store, 'Cyclic', strat)
strat.filters.append(HighPassFilter(adjusted, 0.7))
strat.filters.append(HighPassFilter(cyclic, 0.0))

print("Running base strat...")
strat.run()
print("Generated", strat.trades.count, "trades.")
#print("Applying stops...")
#strat.apply_exit_condition(StopLoss(0.15))
#strat.apply_exit_condition(ReturnTriggeredTrailingStop(0.2, 0.3))
#strat.apply_exit_condition(ReturnTriggeredTrailingStop(0.1, 0.5))

##with open(r'D:\Investing\Workspace\signal_strat.pkl', 'rb') as file:
##    strat = pickle.load(file)

##print("Creating ewmac strat...")
##strat = createEwmacStrategy(store)
##print("Running ewmac strat...")
##strat.run()
##strat.positions = strat.positions.discretise(min_size = 0.7, max_size = 2.0, step = 0.5)

#print("Preparing portfolio...")
#port = Portfolio(strat, 15000)
#port.position_checks.append(PositionCostThreshold(0.02))
#vol_method = StdDevEMA(40)
#volatilities = vol_method(strat.get_indicator_prices()).shift(1)
#from system.core import VolatilitySizingDecorator, FixedNumberOfPositionsSizing
#port.sizing_strategy = VolatilitySizingDecorator(0.2, volatilities, FixedNumberOfPositionsSizing(target_positions = 5))

#print("Running portfolio...")
#port.run_events()
#print("Done...")

print("ready...")
