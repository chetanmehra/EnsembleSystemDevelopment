
# Metrics based on a return series
# Each of these methods expect a pandas Series.

from pandas import Series, DataFrame
from numpy import sign, log, NaN
from sklearn.linear_model import LinearRegression

from data_types.constants import TRADING_DAYS_PER_YEAR

def Sharpe(returns):
    try:
        sharpe = returns.mean() / returns.std()
    except ZeroDivisionError:
        sharpe = NaN
    return sharpe * (TRADING_DAYS_PER_YEAR ** 0.5)

def OptF(returns):
    try:
        optf = returns.mean() / (returns.std() ** 2)
    except ZeroDivisionError:
        optf = NaN
    return optf

def G(returns):
    '''
    The growth rate assuming investment at Optimal F
    '''
    S_sqd = Sharpe(returns) ** 2
    return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1

def GeometricGrowth(returns, N = TRADING_DAYS_PER_YEAR):
    G_base = ((1 + returns.mean()) ** 2 - (returns.std() ** 2))
    G = (abs(G_base) ** 0.5)
    return sign(G_base) * (G ** N) - 1

def K_Ratio(returns):
    lm = LinearRegression(fit_intercept = False)
    returns = returns.dropna()
    returns = ((1 + returns).apply(log)).cumsum()
    X = Series(returns.index).reshape(len(returns), 1)
    lm.fit(X, returns)
    std_error = (Series((lm.predict(X) - returns) ** 2).mean()) ** 0.5
    return ((250 ** 0.5) / len(returns)) * (lm.coef_[0] / std_error)

def Drawdowns(returns):
    '''
    Drawdowns accepts a cumulative returns series, and returns a dataframe with
    columns [Drawdowns, Highwater]
    '''
    high_water = Series(0, index = returns.index, dtype = float)
    for i in range(1, len(returns)):
        high_water[i] = max(high_water[i - 1], returns[i])
    dd = ((returns + 1) / (high_water + 1)) - 1
    return DataFrame({'Drawdown' : dd, 'Highwater' : high_water})