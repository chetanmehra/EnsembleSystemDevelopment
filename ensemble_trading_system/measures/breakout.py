import pandas as pd

from . import SignalElement

class TrailingHighLow(SignalElement):

    def __init__(self, lookback):
        self.period = lookback
        
    def update_param(self, new_params):
        self.period = new_params[0]

    def __call__(self, prices):
        '''
        Returns a panel with the last high, and last low.
        '''
        high = prices.rolling(self.period).max()
        low = prices.rolling(self.period).min()
        return pd.Panel.from_dict({"high" : high, "low" : low})