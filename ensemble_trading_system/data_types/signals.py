'''
Created on 13 Dec 2014

@author: Mark
'''
from system.interfaces import DataElement


class Signal(DataElement):
    '''
    Indicator represent possible trades for the strategy.
    data should be a dataframe of True / False values for determine trade entries & exits
    measures should be a panel of the meaures used to derive the signals.
    '''

    def __init__(self, data, levels, measures):
        self.data = data
        self.measures = measures
        self._levels = levels
    
    @property
    def levels(self):
        if self._levels is None:
            data = self.data.iloc[:, 0]
            data = data[data.notnull()]
            self._levels = set(data)
            self._levels = sorted(self._levels)
        return self._levels

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns
    
    def __getitem__(self, ticker):
        return self.data[ticker]
    
    def __len__(self):
        return len(self.data)

    def plot_measures(self, ticker, start = None, end = None, ax = None):
        self.measures.minor_xs(ticker)[start:end].plot(ax = ax)


    




