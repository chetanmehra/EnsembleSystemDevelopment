
# Metrics based on a return series
# Each of these methods expect a pandas Series.

from pandas import Series, DataFrame
from numpy import sign, log, NaN
from sklearn.linear_model import LinearRegression

def Sharpe(returns):
    try:
        sharpe = returns.mean() / returns.std()
    except ZeroDivisionError:
        sharpe = NaN
    return sharpe

def OptF(returns):
    try:
        optf = returns.mean() / (returns.std() ** 2)
    except ZeroDivisionError:
        optf = NaN
    return optf

def G(returns):
    '''
    G represents the growth in returns assuming investment at optimal F.
    '''
    S_sqd = Sharpe(returns) ** 2
    return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1

def GeometricGrowth(returns, N = 1):
    '''
    The estimated geometric growth taking into consideration the
    variation in returns.
    '''
    G_base = ((1 + returns.mean()) ** 2 - (returns.std() ** 2))
    G = (abs(G_base) ** 0.5)
    return sign(G_base) * (G ** N) - 1

def K_Ratio(returns):
    '''
    K-ratio examines the consistency of an equity's return over time.
    It was developed by Lars Kestner.
    reference: https://www.investopedia.com/terms/k/kratio.asp
    '''
    lm = LinearRegression(fit_intercept = False)
    returns = returns.dropna()
    returns = ((1 + returns).apply(log)).cumsum()
    X = Series(returns.index).reshape(len(returns), 1)
    lm.fit(X, returns)
    std_error = (Series((lm.predict(X) - returns) ** 2).mean()) ** 0.5
    return ((250 ** 0.5) / len(returns)) * (lm.coef_[0] / std_error)


class Drawdowns:

    def __init__(self, returns):
        '''
        Drawdowns accepts a cumulative returns series, and returns a drawdowns
        object with Drawdowns, Highwater, and summary statistics
        '''
        high_water = returns.rolling(len(returns), min_periods = 1).max()
        dd = ((returns + 1) / (high_water + 1)) - 1
        self.data = DataFrame({'Drawdown' : dd, 'Highwater' : high_water})
        self._labels = None

    @property
    def series(self):
        return self.data['Drawdown']

    @property
    def max(self):
        return self.series.min()

    @property
    def highwater(self):
        return self.data['Highwater']

    @property
    def minimums(self):
        return self.series.groupby(self.drawdown_labels).min()[1:]

    @property
    def durations(self):
        return self.series.groupby(self.drawdown_labels).count()[1:]


    def label_drawdowns(self, dd_series):
        below_water = 1 * (dd_series < 0)
        dd_starts = below_water - below_water.shift(1)
        dd_starts.iloc[0] = 0
        dd_starts[dd_starts < 0] = 0
        return dd_starts.cumsum() * below_water

    @property
    def drawdown_labels(self):
        '''
        drawdown_labels assigns a number to each each drawdown. A drawdown will
        then have the same number from start to finish. This can then
        be used in groupby function to produce summary statistics.
        '''
        if self._labels is None:
            self._labels = self.label_drawdowns(self.series)
        return self._labels

    def summary(self):
        summary = Series(dtype = float)
        summary['Max'] = round(-100 * self.max, 2)
        summary['Mean'] = round(-100 * self.minimums.mean(), 2)
        summary['Max duration'] = self.durations.max()
        summary['Avg duration'] = round(self.durations.mean(), 1)
        return summary


class GroupDrawdowns(Drawdowns):
    '''
    GroupDrawdowns provides summary drawdown statistics across a set
    of return series (e.g. market constituents).
    '''

    def __init__(self, returns):
        high_water = returns.rolling(len(returns), min_periods = 1).max()
        dd = ((returns + 1) / (high_water + 1)) - 1
        self.data = {'Drawdown' : dd, 'Highwater' : high_water}
        self.summary = dd.apply(self.get_summary)

    def get_summary(self, series):
        labels = self.label_drawdowns(series)
        minimums = self.get_minimums(series, labels)
        durations = self.get_durations(series, labels)

        summary = Series(dtype = float)
        summary['Max'] = round(-100 * series.min(), 2)
        summary['Mean'] = round(-100 * minimums.mean(), 2)
        summary['Max duration'] = durations.max()
        summary['Avg duration'] = round(durations.mean(), 1)
        return summary

    def get_minimums(self, series, labels):
        return series.groupby(labels).min()[1:]

    def get_durations(self, series, labels):
        return series.groupby(labels).count()[1:]

    @property
    def max(self):
        return self.summary.loc['Max'].max()

    @property
    def mean(self):
        return round(self.summary.loc['Mean'].mean(), 2)

    @property
    def max_duration(self):
        return self.summary.loc['Max duration'].max()

    @property
    def avg_duration(self):
        return round(self.summary.loc['Avg duration'].mean(), 2)


