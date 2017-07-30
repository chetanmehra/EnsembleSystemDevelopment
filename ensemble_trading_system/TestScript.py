'''
Created on 15 Feb 2015

@author: Mark
'''
from multiprocessing import Pool
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from store.file_system import Storage

from data_types.market import Market
from data_types.trades import TradeCollection
from data_types.filter_data import StackedFilterValues, WideFilterValues

from system.core import Strategy, Portfolio
from level_signals import Crossover
from measures.moving_averages import EMA, KAMA
from rules.signal_rules import PositionFromDiscreteSignal

from trade_modifiers.exit_conditions import StopLoss, TrailingStop, ReturnTriggeredTrailingStop
from system.analysis import summary_report


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
#   - Consider giving trades a reference to the market object.
#   - Turn ValueRangeFilter into a BandPassFilter. Ratios for value filters should be pre-calculated.


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

def getMarket(store, source = "ASX", excluded_tickers = None):
    return Market(store.get_instruments(source, excluded_tickers))


def getValues(store, date = None):
    valuations = store.get_valuations(date)
    return StackedFilterValues(valuations.summary, "Valuations")

def getValueRatios(store, type, strat):
    valuation = getValues(store).as_wide_values(type)
    ratios = valuation.value_ratio(strat.get_trade_prices())
    ratios.name = "Value Ratio"
    return ratios

def getValueMetrics(store, date = None):
    metrics = store.get_valuemetrics(date)
    return StackedFilterValues(metrics.summary, "ValuationMetrics")


def signalStratSetup(trade_timing = "CC", ind_timing = "O", params = (120, 50), exchange = "NYSE"):
    store = Storage(exchange)
    strategy = Strategy(trade_timing, ind_timing)
    strategy.market = getMarket(store, exchange)
    strategy.signal_generator = Crossover(slow = EMA(params[0]), fast = EMA(params[1]))
    strategy.position_rules = PositionFromDiscreteSignal(Up = 1)
    return strategy


short_pars = [1, 5, 10, 20, 35, 50]
long_pars = [30, 50, 70, 90, 120, 150, 200]

def test_pars(short_pars, long_pars):
    strat = baseStratSetup()
    summaries = []
    sharpes = pd.DataFrame(None, index = short_pars, columns = long_pars, dtype = float)

    for long in long_pars:
        for short in short_pars:
            strat.measure.update_param((long, short))
            strat.rerun()
            sharpes.loc[short, long] = strat.trades.Sharpe_annual
            summaries.append(((short, long), summary_report(strat.trades)))

    return (sharpes, pd.Dataframe(dict(summaries)))

def strat_summary(pars):
    strat = baseStratSetup(params = pars)
    strat.run()
    attempts = 0
    while True:
        try:
            if attempts >= 2:
                summary = pd.Series(index = ['Sharpe annualised inc slippage'])
            else:
                summary = summary_report(strat.trades)
        except ZeroDivisionError:
            strat.refresh()
            attempts += 1
        else:
            break
            
    return (pars[0], pars[1], summary)

def parallel_test_pars(short_pars, long_pars):
    par_tuples = []
    for long in long_pars:
        short = [s for s in short_pars if s < long]
        par_tuples.extend(zip([long] * len(short), short))
    pool = Pool(processes = 8)
    results = pool.map(strat_summary, par_tuples)

    summaries = []
    sharpes = pd.DataFrame(None, index = long_pars, columns = short_pars, dtype = float)
    for R in results:
        sharpes.loc[R[0], R[1]] = R[2]["Sharpe annualised inc slippage"]
        summaries.append(((R[0], R[1]), R[2]))
    return (sharpes, pd.DataFrame(dict(summaries)))



