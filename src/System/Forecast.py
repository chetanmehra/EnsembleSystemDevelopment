'''
Created on 20 Dec 2014

@author: Mark
'''
import pandas as pd
import numpy as np
from numpy import NaN, mean, std, empty, isinf
from copy import deepcopy
from pandas.core.panel import Panel
from System.Strategy import StrategyContainerElement, ModelElement
from pandas.stats.moments import rolling_mean, rolling_std


class Forecast(StrategyContainerElement):
    
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
    
    @property
    def mean(self):
        return self._mean["Forecast"]
    
    @mean.setter
    def mean(self, data):
        self._mean = data
        
    @property
    def sd(self):
        return self._sd["Forecast"]
    
    @sd.setter
    def sd(self, data):
        self._sd = data
    
    def optF(self):
        optimal_fraction = self.mean / self.sd ** 2
        optimal_fraction[isinf(optimal_fraction)] = 0 
        return optimal_fraction
    
    
class MeanForecastWeighting(object):
    
    def __call__(self, forecasts):
        items = list(bounds(len(forecasts)))
        means = [forecast.mean for forecast in forecasts]
        means = zip(items, means)
        stds = [forecast.sd for forecast in forecasts]
        stds = zip(items, stds)
        means = Panel({item:frame for item, frame in means})
        stds = Panel({item:frame for item, frame in stds})
        return MeanForecast(means.mean(axis = "items"), stds.mean(axis = "items"))
    
    
class MeanForecast(Forecast):
    
    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, data):
        self._mean = data
    
    @property 
    def sd(self):
        return self._sd
    
    @sd.setter
    def sd(self, data):
        self._sd = data


class NullForecaster(ModelElement):
    '''
    The NullForecaster returns 1 for mean where the indicator is equal to any of the
    provided levels in 'in_trade_levels'.
    '''

    def __init__(self, in_trade_levels):
        if not isinstance(in_trade_levels, list):
            in_trade_levels = list(in_trade_levels)
        self.in_trade_levels = in_trade_levels
        self.name = 'Null Forecaster (lvls: {})'.format(','.join(in_trade_levels))

    def execute(self, strategy):
        indicator = strategy.lagged_indicator
        mean = deepcopy(indicator.data)
        mean[:] = 0
        mean = mean.astype(float)
        sd = deepcopy(mean)
        sd[:] = 1
        
        for lvl in self.in_trade_levels:
            mean[indicator.data == lvl] = 1
        
        return Forecast(pd.Panel({"Forecast":mean}), pd.Panel({"Forecast":sd}))
        

class BlockForecaster(ModelElement):
    
    def __init__(self, window):
        self.window = window
        self.name = 'Block Forecast (window: {})'.format(window)
     
    def update_param(self, window):
        self.window = window   
    
    def execute(self, strategy):
        indicator = strategy.lagged_indicator
        returns = strategy.market_returns
        mean_fcst, sd_fcst = self.forecast_mean_sd(indicator, returns)
        return Forecast(mean_fcst, sd_fcst)
    
    def forecast_mean_sd(self, indicator, returns):
        
        tickers = returns.columns
        levels = indicator.levels
        headings = ["Forecast"] + levels
        
        template = empty([len(headings), len(indicator), len(tickers)])
        template[:] = NaN
        mean_fcst = pd.Panel(template.copy(), 
                             items = headings, 
                             major_axis = indicator.index, 
                             minor_axis = tickers)
        sd_fcst = deepcopy(mean_fcst)
        
        for ticker in tickers:
            inds = indicator[ticker]
            rtns = returns[ticker]
            for i, lvl in enumerate(inds):
                j = i + 1
                if j < self.window:
                    pass
                else:
                    ind = inds[(j - self.window):j]
                    rtn = rtns[(j - self.window):j]
                    for level in levels:
                        level_returns = rtn[[ind_val == level for ind_val in ind]]
                        mean_fcst[level][ticker][i] = mean(level_returns)
                        sd_fcst[level][ticker][i] = std(level_returns)
                    mean_fcst["Forecast"][ticker][i] = mean_fcst[lvl][ticker][i]
                    sd_fcst["Forecast"][ticker][i] = sd_fcst[lvl][ticker][i]
        
        return (mean_fcst, sd_fcst)
    

class BlockMeanReturnsForecaster(ModelElement):
    
    def __init__(self, window):
        self.window = window
        self.name = 'Block Mean Rtn Fcst (window: {})'.format(window)
    
    def update_param(self, window):
        self.window = window
                
    def execute(self, strategy):
        returns = strategy.market_returns
        mean_fcst, sd_fcst = self.forecast_mean_sd(returns)
        return Forecast(mean_fcst, sd_fcst)
    
    def forecast_mean_sd(self, returns):
        mean_fcst = pd.Panel({"Forecast":rolling_mean(returns.data, self.window)}) 
        sd_fcst = pd.Panel({"Forecast":rolling_std(returns.data, self.window)})
        
        return (mean_fcst, sd_fcst)

    
    
    
