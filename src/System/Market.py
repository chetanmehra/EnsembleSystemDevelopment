'''
Created on 6 Dec 2014

@author: Mark
'''
import System.Data as Data
from pandas import DateOffset, Panel, DataFrame
from pandas.stats.moments import ewma
import pandas as pd
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import datetime

from System.Position import AverageReturns
from System.Filter import WideFilterValues

class Market(object):
    '''
    System object holds several data frames containing market information for each stock
    it comprises.
    '''

    def __init__(self, tickers, start_date, end_date):
        '''
        Constructor
        '''
        self.start = start_date
        self.end = end_date
        self.downloader = Data.Handler(r"D:\Investing\Data")
        self.instruments = Panel({ticker:DataFrame(None) for ticker in sorted(tickers)})
        
    @property
    def tickers(self):
        return list(self.instruments.items)
    
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

    def download_data(self):
        for ticker in self.tickers:
            raw = self.downloader.get(ticker, self.start, self.end)
            self[ticker] = self.downloader.adjust(raw)

    def load_data(self):
        for ticker in self.tickers:
            self[ticker] = self.downloader.load(ticker, self.start, self.end)
            
    def returns(self, indexer):
        returns = indexer.marketReturns(self)
        return AverageReturns(returns, indexer)

    def volatility(self, window):
        vol = self.close.rolling(window).std()
        return WideFilterValues(vol, "volatility")

    def relative_performance(self, indexer, period):
        returns = indexer.market_returns(self)
        relative = returns.subtract(returns.mean(axis = 'columns'), axis = 'rows')
        return WideFilterValues(ewma(relative, span = period), "relative_return")


    def candlestick(self, ticker, start = None, end = None):
        data = self[ticker][start:end]
        quotes = zip(date2num(data.index.astype(datetime.date)), data.Open, data.High, data.Low, data.Close)
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, quotes)
        ax.xaxis_date()
        fig.autofmt_xdate()
        return (fig, ax)