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

from system.interfaces import DataElement
from data_types.returns import AverageReturns

class Market:
    '''
    Market holds several data frames containing market information for each stock
    it comprises.
    '''

    def __init__(self, store, excluded_tickers = None):
        '''
        Constructor
        '''
        instruments = store.get_instruments(excluded_tickers)
        self.store = store
        self.start = instruments.start
        self.end = instruments.end
        self.exchange = instruments.exchange
        self.instruments = instruments.data
        
    @property
    def tickers(self):
        return list(self.instruments.items)

    @property
    def index(self):
        return self.instruments[self.instruments.items[0]].index
    
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
        return DataFrame(fill_data, index = self.index, columns = self.tickers, dtype = data_type)

    @property
    def open(self):
        return Prices(self._get_series("Open"), ["open"])
    
    @property
    def high(self):
        return Prices(self._get_series("High"), ["close"])
    
    @property
    def low(self):
        return Prices(self._get_series("Low"), ["close"])
    
    @property
    def close(self):
        return Prices(self._get_series("Close"), ["close"])
    
    @property
    def volume(self):
        return Prices(self._get_series("Volume"), ["close"])

    def at(self, timing):
        """
        This method is an implementation of the DataElement.at method, however in this case
        it is used to select the right time series, rather than lag period.
        """
        if timing.target == "O":
            return self.open
        elif timing.target == "C":
            return self.close
    
    def _get_series(self, name):
        return self.instruments.minor_xs(name)

    def returns(self, timing = "close"):
        trade_days = getattr(self, timing)
        return trade_days.returns()

    def candlestick(self, ticker, start = None, end = None):
        data = self[ticker][start:end]
        quotes = zip(date2num(data.index.astype(datetime.date)), data.Open, data.High, data.Low, data.Close)
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, quotes)
        ax.xaxis_date()
        fig.autofmt_xdate()
        return (fig, ax)

    def get_valuations(self, type, sub_type, date = None):
        values = self.store.get_valuations(type, date)
        try:
            fundamentals_data = values.as_wide_values(sub_type, index = self.index)
        except TypeError as E:
            print(E)
        else:
            return Fundamentals(fundamentals_data, sub_type)


class Prices(DataElement):
    """
    A Prices object holds a particular series of prices, e.g. Open.
    """
    def __init__(self, data, calculation_timing, lag = 0):
        self.data = data
        self.calculation_timing = calculation_timing
        self.lag = lag

    def returns(self):
        return AverageReturns((self.data / self.data.shift(1)) - 1)


class Fundamentals(DataElement):
    """
    Provides an interface for different fundamentals data; valuations, 
    earnings, etc.
    """
    def __init__(self, data, name):
        self.name = name
        self.data = data
        self.calculation_timing = ['close']
        self.lag = 0

