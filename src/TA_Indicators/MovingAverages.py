from TA_Indicators.Volatility import EfficiencyRatio
from pandas import DataFrame

def EMA(price_series, period):
        alpha = 2 / (period + 1)
        ema = price_series.copy()
        for i in bounds(len(price_series.index))[1:]:
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


def KAMA(prices, period, fast = 2, slow = 30):
    fastest = 2 / (fast + 1.0)
    slowest = 2 / (slow + 1.0)
    ER = EfficiencyRatio(prices, period)
    sc = (ER * (fastest - slowest) + slowest) ** 2
    kama = DataFrame(None, index = prices.index, columns = prices.columns, dtype = float)
    kama.iloc[period] = prices.iloc[period]
    for i in range((period + 1), len(kama)):
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
