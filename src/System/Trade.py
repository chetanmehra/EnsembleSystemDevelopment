
from pandas import Series, DataFrame, qcut, cut, concat
from numpy import sign, log, hstack
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Factory methods
def create_trades(position_data, strategy):
    prices = strategy.get_trade_prices()
    trades = []
    position_sign = (position_data > 0) * 1
    position_sign[position_data < 0] = -1
    flags = position_sign - position_sign.shift(1)
    # clear missing values from first rows
    start_row = 0
    while all(flags.ix[start_row].isnull()):
        flags.ix[start_row] = 0
        start_row += 1
    # Add trade entries occuring on first day
    flags.ix[start_row][position_sign.ix[start_row] != 0] = position_sign.ix[start_row][position_sign.ix[start_row] != 0]
    for ticker in flags:
        ticker_flags = flags[ticker]
        entries = ticker_flags.index[ticker_flags > 0]
        i = 0
        while i < len(entries):
            entry_day = entries[i]
            i += 1
            if i < len(entries):
                next_entry = entries[i]
            else:
                next_entry = None
            exit_day = ticker_flags[entry_day:next_entry].index[ticker_flags[entry_day:next_entry] < 0]
            if len(exit_day) == 0:
                exit_day = ticker_flags.index[-1]
            else:
                exit_day = exit_day[0]
            trades.append(Trade(ticker, entry_day, exit_day, prices[ticker], position_data[ticker]))
    return TradeCollection(trades)


class TradeCollection(object):

    def __init__(self, data):
        self.tickers = list(set([trade.ticker for trade in data]))
        self.tickers.sort()
        self.trades = data
        self.slippage = 0.011
        self._returns = None
        self._durations = None
        self._daily_returns = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return [trade for trade in self.trades if trade.ticker == key]
        else:
            return self.trades[key]
            

    def as_list(self):
        return self.trades

    def as_dataframe(self):
        data = [trade.as_tuple() for trade in self.trades]
        return DataFrame(data, columns = self.trades[0].cols)

    def as_dataframe_with(self, *args):
        '''
        requires that values is a FilterValues object.
        '''
        df = self.as_dataframe()
        for filter in args:
            self.add_filter_to_df(df, filter)
        return df

    def add_filter_to_df(self, df, filter):
        cols = filter.types
        for col in cols:
            df[col] = None
        if len(cols) == 1:
            cols = cols[0]
        for i in df.index:
            df.loc[i, cols] = filter.get_for_df(df, i)
        return df

    @property
    def count(self):
        return len(self.trades)

    @property
    def returns(self):
        if self._returns is None:
            self._returns = Series([trade.base_return for trade in self.trades])
        return self._returns

    @property
    def returns_slippage(self):
        return self.returns - self.slippage

    @property
    def durations(self):
        if self._durations is None:
            self._durations = Series([trade.duration for trade in self.trades])
        return self._durations

    @property
    def MAEs(self):
        return [trade.MAE for trade in self.trades]

    @property
    def MFEs(self):
        return [trade.MFE for trade in self.trades]

    @property
    def ETDs(self):
        '''
        End Trade Drawdown is defined as the difference between the MFE and the end return
        '''
        return [trade.MFE - trade.base_return for trade in self.trades]

    @property
    def mean_return(self):
        return self.returns.mean()

    @property
    def std_return(self):
        return self.returns.std()

    @property
    def Sharpe(self):
        return self.mean_return / self.std_return

    @property
    def Sharpe_annual(self):
        returns = self.daily_returns
        return (250 ** 0.5) * (returns.mean() / returns.std())

    @property
    def Sharpe_annual_slippage(self):
        returns = self.daily_returns
        total_slippage = self.count * self.slippage
        slippage_per_day = total_slippage / len(returns)
        returns = returns - slippage_per_day
        return (250 ** 0.5) * (returns.mean() / returns.std())

    @property
    def G(self):
        # Refer research note: EQ-2011-003
        S_sqd = self.Sharpe ** 2
        return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1

    @property
    def G_annual(self):
        S_sqd = self.Sharpe_annual ** 2
        return ((1 + S_sqd) ** 2 - S_sqd) ** 0.5 - 1

    @property
    def max_duration(self):
        return max(self.durations)

    @property
    def max_MAE(self):
        return min(self.MAEs)

    @property
    def max_MFE(self):
        return max(self.MFEs)

    def consecutive_wins_losses(self):
        trade_df = self.as_dataframe().sort(columns = 'exit')
        win_loss = sign(trade_df.base_return)
        # Create series which has just 1's and 0's
        positive = Series(hstack(([0], ((win_loss > 0) * 1).values, [0])))
        negative = Series(hstack(([0], ((win_loss < 0) * 1).values, [0])))
        pos_starts = positive.where(positive.diff() > 0)
        pos_starts = Series(pos_starts.dropna().index.tolist())
        pos_ends = positive.where(positive.diff() < 0)
        pos_ends = Series(pos_ends.dropna().index.tolist())
        positive_runs = pos_ends - pos_starts
        neg_starts = negative.where(negative.diff() > 0)
        neg_starts = Series(neg_starts.dropna().index.tolist())
        neg_ends = negative.where(negative.diff() < 0)
        neg_ends = Series(neg_ends.dropna().index.tolist())
        negative_runs = neg_ends - neg_starts
        return (positive_runs, negative_runs)

    def find(self, condition):
        '''
        find accepts a lambda expression which must accept a Trade object as its input and return True/False.
        A TradeCollection of trades meeting the condition is returned.
        '''
        trades = [trade for trade in self.trades if condition(trade)]
        return TradeCollection(trades)
        
    def trade_frame(self, compacted = True, cumulative = True):
        '''
        Returns a dataframe of daily cumulative return for each trade.
        Each row is a trade, and columns are days in trade.
        '''
        df = DataFrame(None, index = range(self.count), columns = range(self.max_duration), dtype = float)
        for i, trade in enumerate(self.trades):
            df.loc[i] = trade.normalised
        if not cumulative:
            df = ((df + 1).T / (df + 1).T.shift(1)).T - 1
        if compacted and df.shape[1] > 10:
            cols = [(11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, 200)]
            trade_df = df.loc[:, 1:10]
            trade_df.columns = trade_df.columns.astype(str)
            for bounds in cols:
                if (df.shape[1] <= bounds[1]):
                    label = '{}+'.format(bounds[0])
                    trade_df[label] = df.loc[:, bounds[0]:].mean(axis = 1)
                    break
                else:
                    label = '{}-{}'.format(*bounds)
                    trade_df[label] = df.loc[:, bounds[0]:bounds[1]].mean(axis = 1)
            final_bound = cols[-1][1]
            if df.shape[1] > final_bound:
                label = '{}+'.format(final_bound + 1)
                trade_df[label] = df.loc[:, (final_bound + 1):].mean(axis = 1)
            return trade_df
        else:
            return df

    @property
    def daily_returns(self):
        '''
        Returns an unsorted and unorderd list of daily returns for all trades.
        Used for calculating daily or annualised statistics.
        '''
        if self._daily_returns is None:
            daily = self.trade_frame(compacted = False, cumulative = False)
            returns = []
            for col in daily:
                returns.extend(daily[col].tolist())
            returns = Series(returns)
            self._daily_returns = returns.dropna()
        return self._daily_returns

    def apply_trailing_stop(self, strat, stop):
        return self.apply_stop(strat, stop, 'apply_trailing_stop')

    def apply_stop_loss(self, strat, stop):
        return self.apply_stop(strat, stop, 'apply_stop_loss')

    def apply_stop(self, strat, stop, type):
        prices = strat.get_trade_prices()
        stopped_trades = []
        for T in self.as_list():
            stopped_trades.append(T.__getattribute__(type)(stop, prices))
        return TradeCollection(stopped_trades)

    def apply_delay(self, strat, delay):
        prices = strat.get_trade_prices()
        delayed_trades = []
        for T in self.as_list():
            new_T = T.apply_delay(delay, prices)
            if new_T is not None:
                delayed_trades.append(new_T)
        return TradeCollection(delayed_trades)

    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_normalised()

    def hist(self, **kwargs):
        # First remove NaNs
        returns = self.returns.dropna()
        plt.hist(returns, **kwargs)

    def plot_MAE(self):
        MAEs = self.MAEs
        returns = self.returns
        x_range = (min(MAEs) -0.05, 0.05)
        y_range = (min(returns) - 0.05, max(returns) + 0.05)
        plt.scatter(self.MAEs, self.returns)
        plt.plot((0, 0), y_range, color = 'black')
        plt.plot(x_range, (0, 0), color = 'black')
        plt.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        plt.xlabel('MAE')
        plt.ylabel('Return')
        plt.xlim(x_range)
        plt.ylim(y_range)

    def plot_MFE(self):
        MFEs = self.MFEs
        returns = self.returns
        x_range = (-0.05, max(MFEs) + 0.05)
        y_range = (min(returns) - 0.05, max(returns) + 0.05)
        plt.scatter(MFEs, returns)
        plt.plot((0, 0), y_range, color = 'black')
        plt.plot(x_range, (0, 0), color = 'black')
        plt.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        plt.xlabel('MFE')
        plt.ylabel('Return')
        plt.xlim(x_range)
        plt.ylim(y_range)


class Trade(object):

    def __init__(self, ticker, entry_date, exit_date, prices, position_size = 1.0):
        self.ticker = ticker
        self.entry = entry_date
        self.exit = exit_date
        self.entry_price = self.get_price(entry_date, prices)
        self.exit_price = self.get_price(exit_date, prices)

        prices = prices[self.entry:self.exit]
        daily_returns = Series(prices / prices.shift(1)) - 1
        daily_returns[0] = (prices[0] / self.entry_price) - 1
        if isinstance(position_size, int) or isinstance(position_size, float):
            self.daily_returns = daily_returns * position_size
        else:
            self.daily_returns = daily_returns * position_size[self.entry:self.exit]
        self.normalised = Series((log(1 + self.daily_returns).cumsum()).values)
        # Note: duration is measured by days-in-market not calendar days
        self.duration = len(self.normalised)
        self.cols = ["ticker", "entry", "exit", "entry_price", "exit_price", "base_return", "duration"]

    def __str__(self):
        first_line = '{0:^23}\n'.format(self.ticker)
        second_line = '{0:10} : {1:10} ({2} days)\n'.format(str(self.entry.date()), str(self.exit.date()), self.duration)
        third_line = '{0:^10.2f} : {1:^10.2f} ({2:.1f} %)\n'.format(self.entry_price, self.exit_price, self.base_return * 100)
        return first_line + second_line + third_line

    def get_price(self, date, prices):
        price = prices[date]
        #if isnull(price):
        #    price = prices[prices.index >= date].dropna()[0]
        return price

    def plot_normalised(self):
        f, axarr = plt.subplots(2, 1, sharex = True)
        axarr[0].set_ylabel('Return')
        axarr[1].set_ylabel('Drawdown')
        axarr[1].set_xlabel('Days in trade')
        self.normalised.plot(ax = axarr[0])
        dd = self.drawdowns()
        dd.Highwater.plot(ax = axarr[0], color = 'red')
        dd.Drawdown.plot(ax = axarr[1])

    def as_tuple(self):
        return tuple(getattr(self, name) for name in self.cols)

    @property
    def base_return(self):
        return (self.exit_price / self.entry_price) - 1

    @property
    def annualised_return(self):
        return (sum(self.normalised.apply(log))) ** (260 / self.duration) - 1

    @property
    def MAE(self):
        return min(self.normalised)

    @property
    def MFE(self):
        return max(self.normalised)

    def drawdowns(self):
        norm = self.normalised
        high_water = Series(0, index = norm.index, dtype = float)
        for i in range(1, len(norm)):
            high_water[i] = max(high_water[i - 1], norm[i])
        dd = ((norm + 1) / (high_water + 1)) - 1

        return DataFrame({'Drawdown' : dd, 'Highwater' : high_water})

    def apply_trailing_stop(self, stop, prices):
        dd = self.drawdowns()
        return self.apply_stop(dd.Drawdown, stop, prices)

    def apply_stop_loss(self, stop, prices):
        return self.apply_stop(self.normalised, stop, prices)

    def apply_stop(self, stop_measure, stop, prices):
        stop = -1 * abs(stop)
        limit_hits = stop_measure.index[stop_measure <= stop]
        if len(limit_hits):
            stopped_day = min(limit_hits)
            stopped_date = prices[self.entry:].index[stopped_day]
            return Trade(self.ticker, self.entry, stopped_date, prices[self.ticker])
        else:
            return self

    def apply_delay(self, delay, prices):
        if self.duration > delay:
            entry = prices[self.entry:].index[delay]
            return Trade(self.ticker, entry, self.exit, prices[self.ticker])
        else:
            return None




