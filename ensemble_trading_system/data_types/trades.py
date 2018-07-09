
from pandas import Series, DataFrame, qcut, cut, concat
from numpy import sign, log, exp, hstack, NaN
import matplotlib.pyplot as plt


from system.metrics import Drawdowns
from data_types import Collection, CollectionItem
from data_types.returns import Returns
from data_types.constants import TRADING_DAYS_PER_YEAR


class TradeCollection(Collection):

    def __init__(self, items):
        self.tickers = list(set([trade.ticker for trade in items]))
        self.tickers.sort()
        self.items = items
        self.slippage = 0.011
        self._returns = None
        self._durations = None
        self._daily_returns = None

    def copy_with(self, items):
        return TradeCollection(items)

    def as_dataframe_with(self, *args):
        '''
        requires that args are a DataElement or a 
        dataframe with dates on index and columns of tickers.
        '''
        df = self.as_dataframe()
        for arg in args:
            self.add_to_df(df, arg)
        return df

    def add_to_df(self, df, data_element):
        df[data_element.name] = NaN
        for i in df.index:
            ticker = df.loc[i].ticker
            entry = df.loc[i].entry
            try:
                df.loc[i, data_element.name] = data_element.loc[entry, ticker]
            except KeyError:
                # Leave this entry blank.
                pass
        return df

    @property
    def returns(self):
        if self._returns is None:
            self._returns = Series([trade.weighted_return for trade in self.items])
        return self._returns

    @property
    def returns_slippage(self):
        return self.returns - self.slippage

    @property
    def durations(self):
        if self._durations is None:
            self._durations = Series([trade.duration for trade in self.items])
        return self._durations

    @property
    def MAEs(self):
        return [trade.MAE for trade in self.items]

    @property
    def MFEs(self):
        return [trade.MFE for trade in self.items]

    @property
    def ETDs(self):
        '''
        End Trade Drawdown is defined as the difference between the MFE and the end return
        '''
        return [trade.MFE - trade.weighted_return for trade in self.items]

    @property
    def mean_return(self):
        return self.returns.mean()

    @property
    def std_return(self):
        return self.returns.std()

    @property
    def Sharpe(self):
        try:
            sharpe = self.mean_return / self.std_return
        except ZeroDivisionError:
            sharpe = NaN
        return sharpe

    @property
    def Sharpe_annual(self):
        returns = self.daily_returns
        return Returns(returns).sharpe()

    @property
    def Sharpe_annual_slippage(self):
        returns = self.daily_returns
        total_slippage = self.count * self.slippage
        slippage_per_day = total_slippage / len(returns)
        returns = returns - slippage_per_day
        return Returns(returns).sharpe()

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
        '''
        Calculates the positive and negative runs in the trade series.
        '''
        trade_df = self.as_dataframe().sort_values(by = 'exit')
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

        
    def trade_frame(self, compacted = True, cumulative = True):
        '''
        Returns a dataframe of daily cumulative return for each trade.
        Each row is a trade, and columns are days in trade.
        '''
        df = DataFrame(None, index = range(self.count), columns = range(self.max_duration), dtype = float)
        for i, trade in enumerate(self.items):
            df.loc[i] = trade.cumulative
        if not cumulative:
            df = ((df + 1).T / (df + 1).T.shift(1)).T - 1
        if compacted and df.shape[1] > 10:
            cols = [(11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, 200)]
            trade_df = df.loc[:, 1:10]
            trade_df.columns = trade_df.columns.astype(str)
            for bounds in cols:
                if df.shape[1] <= bounds[1]:
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

    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_cumulative()

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


# TODO Instead of Trade holding prices, consider retaining a reference to the Market
class Trade(CollectionItem):

    def __init__(self, ticker, entry_date, exit_date, prices, position_size = 1.0):
        self.ticker = ticker
        self.entry = entry_date
        self.exit = exit_date
        self.entry_price = self.get_price(entry_date, prices)
        self.exit_price = self.get_price(exit_date, prices)
        self._position_size = position_size

        self.prices = prices[self.entry:self.exit]
        daily_returns = (self.prices / self.prices.shift(1)) - 1
        daily_returns[0] = (self.prices[0] / self.entry_price) - 1
        self.base_returns = Returns(daily_returns)
        self.calculate_weighted_returns()
        # Fields to use when converting to a tuple
        self.tuple_fields = ["ticker", "entry", "exit", "entry_price", "exit_price", "base_return", "duration", "annualised_return", "normalised_return"]

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

    @property
    def date(self):
        '''
        Supports searching the collection by date.
        '''
        return self.entry

    @property
    def duration(self):
        """
        Duration is measured by days-in-market not calendar days
        """
        return len(self.cumulative)

    @property
    def position_size(self):
        return self._position_size

    @position_size.setter
    def position_size(self, size):
        self._position_size = size
        self.calculate_weighted_returns()

    @property
    def base_return(self):
        return self.base_returns.final()

    @property
    def weighted_return(self):
        return self.weighted_returns.final()

    @property
    def cumulative(self):
        return Series(self.weighted_returns.cumulative().values)

    @property
    def annualised_return(self):
        return self.weighted_returns.annualised()

    @property
    def normalised_return(self):
        return self.weighted_returns.normalised()

    @property
    def MAE(self):
        # Maximum adverse excursion
        return min(self.cumulative)

    @property
    def MFE(self):
        # Maximum favourable excursion
        return max(self.cumulative)

    @property
    def ETD(self):
        # End trade drawdown
        return (1 + self.base_return) / (1 + self.MFE) - 1

    def drawdowns(self):
        # Note we want to return drawdowns with integer index, not
        # date indexed
        return self.weighted_returns.int_indexed().drawdowns()

    def calculate_weighted_returns(self):
        # Short returns are the inverse of daily returns, hence the exponent to the sign of position size.
        base_returns = self.base_returns.data
        position_size = self.position_size
        if isinstance(position_size, int) or isinstance(position_size, float):
            weighted_returns = (1 + (base_returns * abs(position_size))) ** sign(self.position_size) - 1
        elif isinstance(position_size, Series):
            position_size = position_size.shift(1)[self.entry:self.exit]
            weighted_returns = (1 + (base_returns * abs(position_size))) ** sign(position_size[1]) - 1
        else:
            weighted_returns = None
        self.weighted_returns = Returns(weighted_returns)

    # Trade modifiers
    # TODO - revise_entry and revise_exit could be parameter setters
    def revise_entry(self, entry_day):
        '''
        revise_entry accepts an entry_day (integer), and readjusts the trade to
        start on the new (later) entry day.
        '''
        if self.duration > entry_day:
            new_entry = self.base_returns.index[entry_day]
            revised = Trade(self.ticker, new_entry, self.exit, self.prices, self.position_size)
            return revised
        else:
            return None

    def revise_exit(self, exit_day):
        '''
        revise_exit accepts an exit_day (integer), and readjusts the trade to
        end on the new exit day.
        '''
        if exit_day > 0:
            new_exit = self.base_returns.index[exit_day]
            revised = Trade(self.ticker, self.entry, new_exit, self.prices, self.position_size)
            return revised
        else:
            return None

    # Plotting
    def plot_result(self):
        f, axarr = plt.subplots(2, 1, sharex = True)
        axarr[0].set_ylabel('Return')
        axarr[1].set_ylabel('Drawdown')
        axarr[1].set_xlabel('Days in trade')
        self.cumulative.plot(ax = axarr[0])
        self.plot_highwater(ax = axarr[0], color = 'red')
        self.plot_drawdowns(ax = axarr[1])

    def plot_drawdowns(self, **kwargs):
        self.drawdowns().Drawdown.plot(**kwargs)

    def plot_highwater(self, **kwargs):
        self.drawdowns().Highwater.plot(**kwargs)
