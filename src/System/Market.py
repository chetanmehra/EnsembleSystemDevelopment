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
        instruments[key] = self.clean_adj_close(val)
        self.instruments = Panel.from_dict(instruments)
    
    def __getitem__(self, key):
        return self.instruments[key]

    def clean_adj_close(self, instrument, ticker):
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
                end = adj_ratios[adj_ratios < (1 / limit)].index[0]
            except IndexError:
                print(ticker)
                possible_errors[start] = False
            else:
                if (1 / limit) < close_ratios[end] < limit:
                    # Indicates Close is out of sync with Adj Close
                    instrument["Adj Close"][start:(end - 1)] = instrument["Adj Close"][start:(end - 1)] / round(adj_ratios[start])
                    adj_ratios = instrument["Adj Close"] / instrument["Adj Close"].shift(1)
                    possible_errors = adj_ratios > limit
                else:
                    # may be a genuine spike in the data
                    possible_errors[start] = False
        return instrument

    
    @property
    def open(self):
        return self.get_series("Open") * self.adjustment_ratios()
    
    @property
    def high(self):
        return self.get_series("High") * self.adjustment_ratios()
    
    @property
    def low(self):
        return self.get_series("Low") * self.adjustment_ratios()
    
    @property
    def close(self):
        return self.get_series("Adj Close")
    
    @property
    def volume(self):
        return self.get_series("Volume")
    
    def get_series(self, name):
        return self.instruments.minor_xs(name)

    def adjustment_ratios(self):
        return self.get_series("Adj Close") / self.get_series("Close")
    
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
    
        
