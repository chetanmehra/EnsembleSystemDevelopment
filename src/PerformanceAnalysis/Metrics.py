
# Metrics based on a return series
# Each of these methods expect a pandas Series.

from pandas import Series
from sklearn.linear_model import LinearRegression

def Sharpe(returns):
    return returns.mean() / returns.std()

def OptF(returns):
    return returns.mean() / (returns.std() ** 2)

def G(returns):
    S_sqd = Sharpe(returns) ** 2
    return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1

def K_Ratio(returns):
    lm = LinearRegression(fit_intercept = False)
    returns = returns.dropna()
    returns = ((1 + returns).apply(log)).cumsum()
    X = Series(returns.index).reshape(len(returns), 1)
    lm.fit(X, returns)
    std_error = (Series((lm.predict(X) - returns) ** 2).mean()) ** 0.5
    return ((250 ** 0.5) / len(returns)) * (lm.coef_[0] / std_error)
