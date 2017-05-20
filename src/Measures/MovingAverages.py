from Indicators.Volatility import EfficiencyRatio
from pandas import DataFrame

class MovingAverage(object):

    def __call__(self, prices):
        raise NotImplementedError()

    def update_param(self, new_params):
        raise NotImplementedError()


class EMA(MovingAverage):

    def __init__(self, period):
        self.period = period
        self.name = 'EMA{}'.format(period)
    
    def __call__(self, prices):
        return prices.ewm(span = self.period).mean()

    def update_param(self, new_params):
        self.period = new_params



class KAMA(MovingAverage):

    def __init__(self, period, fast = 2, slow = 30):
        self.fast = fast
        self.slow = slow
        self.period = period
        self.name = 'KAMA.{}.{}.{}'.format(period, fast, slow)

    def __call__(self, prices):
        fastest = 2 / (self.fast + 1.0)
        slowest = 2 / (self.slow + 1.0)
        ER = EfficiencyRatio(prices, self.period)
        sc = (ER * (fastest - slowest) + slowest) ** 2
        kama = DataFrame(None, index = prices.index, columns = prices.columns, dtype = float)
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