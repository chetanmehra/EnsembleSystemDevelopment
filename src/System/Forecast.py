'''
Created on 20 Dec 2014

@author: Mark
'''
import pandas as pd
import numpy as np
from numpy import NaN, empty, isinf
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
    
    def __init__(self, window, pooled = False, average = "mean"):
        self.window = window
        self.name = 'Block Forecast (window: {})'.format(window)
        self.pooled = pooled
        self.average = average
     
    def update_param(self, window):
        self.window = window   
    
    def execute(self, strategy):
        indicator = strategy.indicator
        returns = strategy.market_returns
        if self.pooled:
            mean_fcst, sd_fcst = self.pooled_forecast_mean_sd(indicator, returns, self.average)
        else:
            mean_fcst, sd_fcst = self.forecast_mean_sd(indicator, returns, self.average)
        return Forecast(mean_fcst, sd_fcst)
    
    def forecast_mean_sd(self, indicator, returns, average):
        
        tickers = returns.columns
        levels = indicator.levels
        headings = ["Forecast"] + levels
        
        mean_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)
        sd_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)
        
        for ticker in tickers:
            # HACK indicator and return lags are hard coded in BlockForecaster
            # indicator for decision is available at the same time of forecasting, no lag
            decision_inds = indicator[ticker]
            # indicator aligned with returns needs to be return lag + 1 (2 total)
            # however, when indexing into dataframe the current day is not returned so 
            # is in effect one day lag built in already. Therefore only shift 1.
            aligned_inds = indicator[ticker].shift(1)
            # returns need to be lagged one day for "CC", "O" timings
            # As above, indexing into dataframe effectively already includes lag of 1.
            rtns = returns[ticker]
            for i, lvl in enumerate(decision_inds):
                if i < self.window:
                    continue
                ind = aligned_inds[(i - self.window):i]
                rtn = rtns[(i - self.window):i]
                for level in levels:
                    level_returns = rtn[ind == level]
                    if average == "mean":
                        mean_fcst[level][ticker][i] = level_returns.mean()
                    else:
                        mean_fcst[level][ticker][i] = level_returns.median()
                    sd_fcst[level][ticker][i] = level_returns.std()
                mean_fcst["Forecast"][ticker][i] = mean_fcst[lvl][ticker][i]
                sd_fcst["Forecast"][ticker][i] = sd_fcst[lvl][ticker][i]
        
        return (mean_fcst, sd_fcst)
 
    
    def pooled_forecast_mean_sd(self, indicator, returns, average):
        
        tickers = returns.columns
        levels = indicator.levels
        headings = ["Forecast"] + levels
        
        mean_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)
        sd_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)
        
        decision_inds = indicator.data
        aligned_inds = indicator.data.shift(1)
        rtns = returns.data
        for i in range(len(decision_inds)):
            if i < self.window:
                continue
            ind = aligned_inds[(i - self.window):i]
            rtn = rtns[(i - self.window):i]
            for level in levels:
                level_returns = rtn[ind == level].unstack()
                if average == "mean":
                    mean_fcst[level][:][i] = level_returns.mean()
                else:
                    mean_fcst[level][:][i] = level_returns.median()
                sd_fcst[level][:][i] = level_returns.std()
            ticker_levels = decision_inds.ix[i]
            for ticker in tickers:
                lvl = ticker_levels[ticker]
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

    
    
    
