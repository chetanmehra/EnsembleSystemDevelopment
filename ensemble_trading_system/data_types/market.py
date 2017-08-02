'''
Created on 6 Dec 2014

@author: Mark
'''
from pandas import DateOffset, Panel, DataFrame
from pandas.stats.moments import ewma
import pandas as pd
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import datetime

# Import from related packages
import os
import sys
sys.path.append(os.path.join("C:\\Users", os.getlogin(), "Source\\Repos\\FinancialDataHandling\\financial_data_handling"))
from formats.price_history import Instruments

from data_types.positions import AverageReturns
from data_types.filter_data import WideFilterValues

class Market(object):
    '''
    System object holds several data frames containing market information for each stock
    it comprises.
    '''

    def __init__(self, instruments):
        '''
        Constructor
        '''
        self.name = instruments.name
        self.start = instruments.start
        self.end = instruments.end
        self.exchange = instruments.exchange
        self.instruments = instruments.data
        
    @property
    def tickers(self):
        return list(self.instruments.items)

    def as_instruments(self):
        instruments = Instruments(self.name)
        instruments.data = self.data
        instruments.start = self.start
        instruments.end = self.end
        instruments.exchange = self.exchange
        return instruments
    
    def __setitem__(self, key, val):
        instruments = dict(self.instruments)
        instruments[key] = val
        self.instruments = Panel.from_dict(instruments)
    
    def __getitem__(self, key):
        return self.instruments[key]

    def get_empty_dataframe(self, fill_data = None):
        if isinstance(fill_data, str):
            data_type = object
        else:
            data_type = float
        return DataFrame(fill_data, index = self.close.index, columns = self.tickers, dtype = data_type)

    @property
    def open(self):
        return self._get_series("Open")
    
    @property
    def high(self):
        return self._get_series("High")
    
    @property
    def low(self):
        return self._get_series("Low")
    
    @property
    def close(self):
        return self._get_series("Close")
    
    @property
    def volume(self):
        return self._get_series("Volume")
    
    def _get_series(self, name):
        return self.instruments.minor_xs(name)

    def returns(self, indexer):
        returns = indexer.market_returns(self)
        return AverageReturns(returns, indexer)

    def candlestick(self, ticker, start = None, end = None):
        data = self[ticker][start:end]
        quotes = zip(date2num(data.index.astype(datetime.date)), data.Open, data.High, data.Low, data.Close)
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, quotes)
        ax.xaxis_date()
        fig.autofmt_xdate()
        return (fig, ax)