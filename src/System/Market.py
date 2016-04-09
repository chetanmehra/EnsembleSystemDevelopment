'''
Created on 6 Dec 2014

@author: Mark
'''
import System.Data as Data
from pandas import Panel, DataFrame
from System.Position import AverageReturns

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
        instruments[key] = val
        self.instruments = Panel.from_dict(instruments)
    
    def __getitem__(self, key):
        return self.instruments[key]
    
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
        return self.get_series("Close")
    
    @property
    def volume(self):
        return self.get_series("Volume")
    
    def get_series(self, name):
        return self.instruments.minor_xs(name)
    
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
    
        
