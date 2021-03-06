
from pandas import DataFrame
import numpy as np

from . import SignalElement
from .volatility import EfficiencyRatio


class EMA(SignalElement):

    def __init__(self, period):
        self.period = period
        
    def __call__(self, prices):
        return prices.ewm(span = self.period).mean()

    def update_param(self, new_params):
        self.period = new_params


class KAMA(SignalElement):
    '''
    Calculates the Kaufman adaptive moving average. The smooth parameter varies between the 
    square of the fast and slow parameters based on the efficiency ratio calculated from 
    the period number of previous days.
    '''
    def __init__(self, period, fast = 2, slow = 30):
        self.fast = fast
        self.slow = slow
        self.period = period
        self.eff_ratio = EfficiencyRatio(period)

    @property
    def name(self):
        return 'KAMA.{}.{}.{}'.format(self.period, self.fast, self.slow)

    def __call__(self, prices):
        fastest = 2 / (self.fast + 1.0)
        slowest = 2 / (self.slow + 1.0)
        ER = self.eff_ratio(prices)
        sc = (ER * (fastest - slowest) + slowest) ** 2
        kama = DataFrame(None, index = prices.index, columns = prices.tickers, dtype = float)
        kama.iloc[self.period] = prices.iloc[self.period]
        for i in range((self.period + 1), len(kama)):
            prev_kama = kama.iloc[(i - 1)]
            curr_prices = prices.iloc[i]
            curr_kama = prev_kama + sc.iloc[i] * (curr_prices - prev_kama)
            missing_prev_kama = prev_kama.isnull()
            if any(missing_prev_kama):
                prev_kama[missing_prev_kama] = curr_prices[missing_prev_kama]
            missing_curr_kama = curr_kama.isnull()
            if any(missing_curr_kama):
                curr_kama[missing_curr_kama] = prev_kama[missing_curr_kama]
            kama.iloc[i] = curr_kama
        return kama

    def update_params(self, new_params):
        self.period = new_params[0]
        self.fast = new_params[1]
        self.slow = new_params[2]


class LinearTrend(SignalElement):
    '''
    Linear trend produces a rolling linear regression output for the given span.
    '''
    def __init__(self, span):
        self.period = span
        self.name = "LinTrend.{}".format(span)
        X = np.asarray(range(1, self.period + 1))
        self.X_bar = X.mean()
        self.X_diff = X - self.X_bar
        self.SSE_X = (self.X_diff ** 2).sum()

    def __call__(self, prices):
        return prices.ffill().rolling(self.period).apply(self.trend)

    def trend(self, Y):
        '''
        This gets called by rolling apply, and gets passed a numpy.ndarray.
        This will return the linear trend at the end of the supplied window.
        '''
        Y_bar = Y.mean()
        Y_diff = Y - Y_bar
        slope = (self.X_diff * Y_diff).sum() / self.SSE_X
        intercept = Y_bar - slope * self.X_bar
        return slope * self.period + intercept


