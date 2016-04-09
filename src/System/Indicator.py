'''
Created on 13 Dec 2014

@author: Mark
'''
from numpy import isnan
from pandas import notnull
from pandas.stats.moments import ewma
from System.Strategy import StrategyContainerElement, MeasureElement

class Indicator(StrategyContainerElement):
    
    def __init__(self, data):
        self._levels = None
        self.data = data
    
    @property
    def levels(self):
        if self._levels is None:
            for ticker in self.data:
                data = self.data[ticker]
                data = data[notnull(data)]
                self._levels = set(data)
            self._levels = sorted(self._levels)
        return self._levels
    
    @property
    def index(self):
        return self.data.index
    
    def __getitem__(self, ticker):
        return self.data[ticker]
    
    def __len__(self):
        return len(self.data)


class Crossover(MeasureElement):
    
    def __init__(self, slow, fast):
        self.fast = fast
        self.slow = slow
        
    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = ewma(prices, span = self.fast)
        slow_ema = ewma(prices, span = self.slow)
        levels = fast_ema > slow_ema
        return Indicator(levels.astype('str'))
    
    def update_param(self, new_params):
        self.slow = new_params[0]
        self.fast = new_params[1]
     
        
def EMA(price_series, period):
        alpha = 2 / (period + 1)
        ema = price_series.copy()
        for i in range(len(price_series.index))[1:]:
            current_prices = price_series.iloc[i]
            previous_ema = ema.iloc[i - 1]
            
            ema.iloc[i] = (1 - alpha) * previous_ema + alpha * current_prices
            
            missing = isnan(ema.iloc[i])
            if any(missing):
                ema.iloc[i][missing] = previous_ema[missing]
            still_missing = isnan(ema.iloc[i])
            if any(still_missing):
                ema.iloc[i][still_missing] = current_prices[still_missing]
                
        return ema        
        
                
        


