
from pandas import rolling_sum

def EfficiencyRatio(prices, period):
    '''
    The Efficiency Ratio is taken from Kaufman's KAMA indicator.
    Refer New Trading Systems and Methods 4th Ed (p732)
    The Efficiency Ratio is the ratio of the price change of a period
    divided by the sum of daily changes.
    prices - expected to be a DataFrame (dates vs tickers)
    period - an integer lookback window.
    '''
    overall_change = prices.diff(period).abs()
    daily_sum = rolling_sum(prices.diff().abs(), period)
    return overall_change / daily_sum