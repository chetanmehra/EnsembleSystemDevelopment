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
        self.downloader = Data.Handler("/home/mark/Data/MarketData/Stocks/Python/")
        self.instruments = Panel({ticker:DataFrame(None) for ticker in sorted(tickers)})
        
    @property
    def tickers(self):
        return list(self.instruments.items)
    
    def __setitem__(self, key, val):
        instruments = dict(self.instruments)
        instruments[key] = self.clean_adj_close(val)
        self.instruments = Panel.from_dict(instruments)
    
    def __getitem__(self, key):
        adjust = self.adjustment_ratios()
        df = self.instruments[key]
        df.Open = df.Open * adjust[key]
        df.High = df.High * adjust[key]
        df.Low = df.Low * adjust[key]
        df.Close = df.Close * adjust[key]
        return df

    def clean_adj_close(self, instrument):
        '''
        Takes a dataframe [OHLCV & Adj Close] for a ticker
        Tries to find any erroneous Adj Close values caused by stock splits.
        '''
        adj_ratios = instrument["Adj Close"] / instrument["Adj Close"].shift(1)
        close_ratios = instrument["Close"] / instrument["Close"].shift(1)
        limit = 3.0
        possible_errors = adj_ratios > limit
        while any(possible_errors):
            try:
                start = adj_ratios[possible_errors].index[0]
                ix = 0
                end = adj_ratios[adj_ratios < (1 / limit)].index[ix]
                while end < start:
                    ix += 1
                    end = adj_ratios[adj_ratios < (1 / limit)].index[ix]
            except IndexError:
                possible_errors[start] = False
            else:
                if (1 / limit) < close_ratios[end] < limit:
                    # Indicates Close is out of sync with Adj Close
                    divisor = round(adj_ratios[start])
                    instrument["Adj Close"][start:(end - DateOffset(1))] = instrument["Adj Close"][start:(end - DateOffset(1))] / divisor
                    adj_ratios = instrument["Adj Close"] / instrument["Adj Close"].shift(1)
                    possible_errors = adj_ratios > limit
                else:
                    # may be a genuine spike in the data
                    possible_errors[start] = False
        return instrument

    def get_empty_dataframe(self):
        return DataFrame(index = self.close.index, columns = self.tickers, dtype = float)


    @property
    def open(self):
        return self.get_series("Open")
    
    @property
    def high(self):
        return self.get_series("High")
    
    @property
    def low(self):
        return self.get_series("Low")
    
    @property
    def close(self):
        return self.get_series("Adj Close")
    
    @property
    def volume(self):
        return self.get_series("Volume")
    
    def get_series(self, name):
        series = self.instruments.minor_xs(name)
        if name not in ["Adj Close", "Volume"]:
            series = series * self.adjustment_ratios()
        return series

    def adjustment_ratios(self):
        return self.get_series("Adj Close") / self.instruments.minor_xs("Close")
    
    @property
    def status(self):
        download_status = {}
        for ticker in self.tickers:
            download_status[ticker] = False if len(self[ticker]) is 0 else True
        return download_status
    
    def download_data(self):
        for ticker in self.tickers:
            self[ticker] = self.downloader.get(ticker + ".AX", self.start, self.end)
            
    def returns(self, indexer):
        returns = indexer.market_returns(self)
        return AverageReturns(returns)

    def volatility(self, window):
        vol = pd.rolling_std(self.close, window)
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