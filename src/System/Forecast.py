'''
Created on 20 Dec 2014

@author: Mark
'''
import pandas as pd
import numpy as np
from numpy import NaN, empty, isinf
from copy import deepcopy
from pandas import Panel
from pandas.stats.moments import rolling_mean, rolling_std
from System.Strategy import StrategyContainerElement, ModelElement



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
        i = self.window
        while i < len(decision_inds):
            if i < self.window:
                continue
            ind = aligned_inds[(i - self.window):i]
            rtn = rtns[(i - self.window):i]
            for level in levels:
                level_returns = rtn[ind == level].unstack()
                if average == "mean":
                    mean_fcst[level].iloc[i, :] = level_returns.mean()
                else:
                    mean_fcst[level].iloc[i, :] = level_returns.median()
                sd_fcst[level].iloc[i, :] = level_returns.std()
            i += 1
        mean_fcst = mean_fcst.fillna(method = 'ffill')
        sd_fcst = sd_fcst.fillna(method = 'ffill')
        for ticker in tickers:
            decision_ind = decision_inds[ticker]
            for i, lvl in enumerate(decision_ind):
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

class SmoothDetrendedForecaster(ModelElement):

    def __init__(self, span):
        self.span = span
        self.name = 'Smooth Detrended Fcst (span: {})'.format(span)

    def update_param(self, span):
        self.smooth = EMA(span)

    def execute(self, strategy):
        returns = strategy.market_returns
        indicator = strategy.indicator
        mean_fcst, sd_fcst = self.forecast_mean_sd(indicator, returns)
        return Forecast(mean_fcst, sd_fcst)

    def forecast_mean_sd(self, indicator, returns):

        tickers = returns.columns
        levels = indicator.levels
        headings = ["Forecast"] + levels + ['trend'] + [L + "_detrend" for L in levels]
        
        measures = {}
        for level in levels:
            measures[level] = pd.DataFrame(NaN, index = indicator.index, columns = tickers)

        mean_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)
        sd_fcst = pd.Panel(NaN, items = headings, major_axis = indicator.index, minor_axis = tickers)

        # HACK indicator and return lags are hard coded in SmoothDetrendedForecaster
        # indicator for decision is available at the same time of forecasting, no lag
        decision_inds = indicator.data
        # indicator aligned with returns needs to be return lag + 1 (2 total)
        # however, when indexing into dataframe the current day is not returned so 
        # is in effect one day lag built in already. Therefore only shift 1.
        aligned_inds = indicator.data.shift(2)
        # returns need to be lagged one day for "CC", "O" timings
        # As above, indexing into dataframe effectively already includes lag of 1.
        returns = returns.data.shift(1)

        for day in decision_inds.index:
            for level in levels:
                level_index = aligned_inds.ix[day] == level
                measures[level].loc[day, level_index] = returns.loc[day, level_index]

        mean_fcst["trend"] = returns.ewm(span = self.span).mean()
        sd_fcst["trend"] = returns.ewm(span = self.span).std()

        for level in levels:
            measures[level] = measures[level].fillna(method = 'ffill')
            mean_fcst[level] = measures[level].ewm(span = self.span).mean()
            mean_fcst[level + "_detrend"] = mean_fcst[level] - mean_fcst["trend"]
            sd_fcst[level] = measures[level].ewm(span = self.span).std()
            sd_fcst[level + "_detrend"] = sd_fcst[level] - sd_fcst["trend"]

        for day in decision_inds.index:
            for level in levels:
                level_index = decision_inds.ix[day] == level
                mean_fcst["Forecast"].loc[day, level_index] = mean_fcst[level + "_detrend"].loc[day, level_index]
                sd_fcst["Forecast"].loc[day, level_index] = sd_fcst[level + "_detrend"].loc[day, level_index]

        mean_fcst["Forecast"] = mean_fcst["Forecast"].fillna(method = 'ffill')
        sd_fcst["Forecast"] = sd_fcst["Forecast"].fillna(method = 'ffill')
        return (mean_fcst, sd_fcst)

        


    
    
