

# TODO Average True Range volatility measure

class EfficiencyRatio:
    '''
    The Efficiency Ratio is taken from Kaufman's KAMA indicator.
    Refer New Trading Systems and Methods 4th Ed (p732)
    The Efficiency Ratio is the ratio of the price change of a period
    divided by the sum of daily changes.
    prices - expected to be a DataFrame (dates vs tickers)
    period - an integer lookback window.
    '''

    def __init__(self, period):
        self.period = period

    def __call__(self, prices):
        overall_change = prices.diff(period).abs()
        daily_sum = prices.diff().abs().rolling(window = period, center = False).sum()
        return overall_change / daily_sum


class StdDevRolling:

    def __init__(self, period):
        self.period = period

    def __call__(self, prices):
        return prices.rolling(span = self.period).std()


class StdDevEMA:

    def __init__(self, period):
        self.period = period

    def __call__(self, prices):
        return prices.ewm(span = self.period).std()