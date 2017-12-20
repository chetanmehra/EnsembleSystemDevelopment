import pandas as pd

class TrailingHighLow:

    def __init__(self, lookback):
        self.lookback = lookback
        self.name = 'Breakout_{}'.format(lookback)
    

    def update_param(self, new_params):
        self.lookback = new_params[0]

    def __call__(self, prices):
        '''
        Returns a panel with the last high, and last low.
        '''
        high = prices.rolling(span = self.lookback).max()
        low = prices.rolling(span = self.lookback).min()
        return pd.Panel.from_dict({"high" : high, "low" : low})