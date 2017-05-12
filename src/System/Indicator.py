'''
Created on 13 Dec 2014

@author: Mark
'''
from numpy import isnan
from pandas import notnull, Panel, DataFrame, Series
from Indicators.MovingAverages import KAMA
from System.Strategy import StrategyContainerElement, MeasureElement

class Indicator(StrategyContainerElement):
    
    def __init__(self, data, measures):
        self._levels = None
        self.data = data
        self.measures = measures
    
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

    @property
    def columns(self):
        return self.data.columns
    
    def __getitem__(self, ticker):
        return self.data[ticker]
    
    def __len__(self):
        return len(self.data)

    def plot_measures(self, ticker, start = None, end = None, ax = None):
        self.measures.minor_xs(ticker)[start:end].plot(ax = ax)



class Crossover(MeasureElement):
    
    def __init__(self, slow, fast):
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return "x".join([self.fast.name, self.slow.name])

    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = self.fast(prices)
        slow_ema = self.slow(prices)
        levels = fast_ema > slow_ema
        return Indicator(levels.astype('str'), Panel.from_dict({'Fast':fast_ema, 'Slow':slow_ema}))
    
    def update_param(self, new_params):
        self.slow.update_param(new_params[0])
        self.fast.update_param(new_params[1])



class TripleCrossover(MeasureElement):
    
    def __init__(self, slow, mid, fast):
        self.fast = fast
        self.mid = mid
        self.slow = slow

    @property
    def name(self):
        return ".".join(["TRPx", str(self.fast), str(self.mid), str(self.slow)])
        
    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = ewma(prices, span = self.fast)
        mid_ema = ewma(prices, span = self.mid)
        slow_ema = ewma(prices, span = self.slow)
        levels = (fast_ema > mid_ema) & (mid_ema > slow_ema)
        return Indicator(levels.astype('str'), Panel.from_dict({'Fast':fast_ema, 'Mid':mid_ema, 'Slow':slow_ema}))
    
    def update_param(self, new_params):
        pars = list(new_params)
        pars.sort()
        self.fast = pars[0]
        self.mid = pars[1]
        self.slow = pars[2]


class ValueWeightedEMA(MeasureElement):

    def __init__(self, values, fast, slow):
        self.values = values
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return ".".join([self.values.name, "Wtd", str(self.fast), str(self.slow)])

    def execute(self, strategy):
        prices = strategy.get_indicator_prices()
        value_ratio = DataFrame(prices)
        value_ratio[:] = None
        for col in self.values:
            value_ratio[col] = self.values[col]
        value_ratio.fillna(method = 'ffill')
        value_ratio = value_ratio / prices



    def update_param(self, new_params):
        pass



class TrendBenchmark(object):
    '''
    The TrendBenchmark is not intended for use in a strategy, but for testing the performance
    of an indicator against ideal, perfect hindsight identification of trends.
    '''
    def __init__(self, period):
        self.period = period
        
    def __call__(self, strategy):
        prices = strategy.get_indicator_prices()
        trend = DataFrame(None, index = prices.index, columns = prices.columns, dtype = float)
        last_SP = Series(None, index = prices.columns)
        current_trend = Series('-', index = prices.columns)
        for i in range(prices.shape[0] - self.period):
            # If there are not any new highs in the recent period then must have been 
            # a swing point high.
            SPH = ~(prices.iloc[(i + 1):(i + self.period)] > prices.iloc[i]).any()
            # NaN in series will produce false signals and need to be removed
            SPH = SPH[prices.iloc[i].notnull()]
            SPH = SPH[SPH]
            # Only mark as swing point high if currently in uptrend or unidentified trend, otherwise ignore.
            SPH = SPH[current_trend[SPH.index] != 'DOWN']
            if not SPH.empty:
                current_trend[SPH.index] = 'DOWN'
                trend.loc[trend.index[i], SPH.index] = prices.iloc[i][SPH.index]
            # Repeat for swing point lows.
            SPL = ~(prices.iloc[(i + 1):(i + self.period)] < prices.iloc[i]).any()
            SPL = SPL[prices.iloc[i].notnull()]
            SPL = SPL[SPL]
            SPL = SPL[current_trend[SPL.index] != 'UP']
            if not SPL.empty:
                current_trend[SPL.index] = 'UP'
                trend.loc[trend.index[i], SPL.index] = prices.iloc[i][SPL.index]
        self.trend = trend.interpolate()



