
from pandas import Series, DataFrame, qcut, cut, concat
from numpy import sign, log, exp, hstack, NaN
import matplotlib.pyplot as plt

from system.metrics import Drawdowns, GroupDrawdowns
from data_types import Collection, CollectionItem
from data_types.returns import Returns


class TradeCollection(Collection):

    def __init__(self, items):
        super().__init__(items)
        self.tickers = list(set([trade.ticker for trade in items]))
        self.tickers.sort()
        self.slippage = 0.011
        # Cache attributes
        self._returns = None
        self._daily_returns = None
        self._durations = None
        self._MAEs = None
        self._MFEs = None
        self._ETDs = None
        self._maxDDs = None
        

    def copy(self, items = None, **kwargs):
        if items is None:
            items = [item.copy() for item in self.items]
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
        if self._MAEs is None:
            self._MAEs = [trade.MAE for trade in self.items]
        return self._MAEs

    @property
    def MFEs(self):
        if self._MFEs is None:
            self._MFEs = [trade.MFE for trade in self.items]
        return self._MFEs

    @property
    def ETDs(self):
        if self._ETDs is None:
            self._ETDs = [trade.ETD for trade in self.items]
        return self._ETDs

    @property
    def max_drawdowns(self):
        if self._maxDDs is None:
            self._maxDDs = [trade.maxDD for trade in self.items]
        return self._maxDDs

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

    def summary(self):
        '''
        Returns a Series with summary trade return statistics
        '''
        winners = self.returns > 0
        losers = self.returns < 0
        summary = Series(dtype = float)
        summary['Number of trades'] = self.count
        summary['Percent winners'] = round(100 * winners.sum() / self.count, 2)
        summary['Number of winners'] = winners.sum()
        summary['Number of losers'] = losers.sum()
        summary['Number even'] = (self.returns == 0).sum()
        summary['Avg return'] = round(100 * self.returns.mean(), 2)
        summary['Avg return inc slippage'] = round(100 * (self.returns - self.slippage).mean(), 2)
        summary['Median return'] = round(100 * self.returns.median(), 2)
        summary['Average winner'] = round(100 * self.returns[winners].mean(), 2)
        summary['Average loser'] = round(100 * self.returns[losers].mean(), 2)
        summary['Win/Loss ratio'] = round(abs(self.returns[winners].mean() / self.returns[losers].mean()), 2)
        summary['Largest winner'] = round(100 * self.returns[winners].max(), 2)
        summary['Largest loser'] = round(100 * self.returns[losers].min(), 2)
        summary['Average duration'] = round(self.durations.mean(), 1)
        summary['Average duration winners'] = round(self.durations[winners].mean(), 1)
        summary['Average duration losers'] = round(self.durations[losers].mean(), 1)
        return summary

    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_cumulative()

    def hist(self, **kwargs):
        # First remove NaNs
        returns = self.returns.dropna()
        plt.hist(returns, **kwargs)

    def plot_MAE(self, **kwargs):
        self._scatterplot(self.MAEs, self.returns, 'MAE', 'Return', **kwargs)

    def plot_MFE(self, **kwargs):
        self._scatterplot(self.MFEs, self.returns, 'MFE', 'Return', **kwargs)

    def plot_ETD_vs_DD(self, **kwargs):
        '''
        Plotting the end trade drawdown vs the max drawdown in the trade will
        give an indication of if a trailing stop will improve the end return
        or just cut the trade early.
        '''
        self._scatterplot(self.ETDs, self.max_drawdowns, 'ETD', 'Max DD', **kwargs)

    def plot_MAE_vol_adjusted(self, volatility, **kwargs):
        if not hasattr(volatility, 'name'):
            volatility.name = 'volatility'
        trade_df = self.as_dataframe_with(volatility)
        MAE_Adjusted = self.MAEs / trade_df[volatility.name]
        self._scatterplot(MAE_Adjusted, self.returns, 'MAE adj.', 'Return', **kwargs)

    def _scatterplot(self, X, Y, xlabel, ylabel, **kwargs):
        if 'x_range' not in kwargs:
            x_range = (min(0, min(X) - 0.01), max(X) + 0.01)
        else:
            x_range = kwargs.pop('x_range')
        if 'y_range' not in kwargs:
            y_range = (min(Y) - 0.01, max(Y) + 0.01)
        else:
            y_range = kwargs.pop('y_range')
        if 'ax' not in kwargs:
            fig, ax = plt.subplots(1)
        else:
            ax = kwargs.pop('ax')
        
        ax.plot(X, Y, '.', **kwargs) # produces a point scatter plot
        ax.plot((0, 0), y_range, color = 'black')
        ax.plot(x_range, (0, 0), color = 'black')
        ax.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        return ax

class Trade(CollectionItem):

    def __init__(self, entry_event, exit_event, position_loc, base_returns_loc, returns_loc):
        self.ticker = entry_event.ticker
        self.entry = entry_event.date
        self.returns_start = entry_event.offset(1)
        self.positions_end = exit_event.offset(-1)
        self.exit = exit_event.date
        self.entry_event = entry_event
        self.exit_event = exit_event
        self._position_loc = position_loc
        self._base_returns = base_returns_loc
        self.weighted_returns = returns_loc
        # self.calculate_weighted_returns()
        # Fields to use when converting to a tuple
        self.tuple_fields = ["ticker", "entry", "exit", "duration", "weighted_return", "base_return", "annualised_return", "normalised_return"]

    def __str__(self):
        values = list(self.as_tuple())
        for i, v in enumerate(values):
            if isinstance(v, float):
                values[i] = round(v, 3)
        return Series(values, index = self.tuple_fields).__str__()

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
        self._position_loc[self.entry:self.positions_end, self.ticker]

    @position_size.setter
    def position_size(self, size):
        self._position_loc[self.entry:self.positions_end, self.ticker] = size

    @property
    def base_returns(self):
        return Returns(self._base_returns[self.returns_start:self.exit, self.ticker])

    @property
    def base_return(self):
        return self.base_returns.final()

    @property
    def weighted_returns(self):
        return Returns(self._weighted_returns[self.returns_start:self.exit, self.ticker])

    @weighted_returns.setter
    def weighted_returns(self, returns_loc):
        self._weighted_returns = returns_loc

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
        return (1 + self.weighted_return) / (1 + self.MFE) - 1

    @property
    def maxDD(self):
        return self.drawdowns().max

    def drawdowns(self):
        # Note we want to return drawdowns with integer index, not
        # date indexed
        return self.weighted_returns.int_indexed().drawdowns()

    # def calculate_weighted_returns(self):
    #     # Short returns are the inverse of daily returns, hence the exponent to the sign of position size.
    #     base_returns = self.base_returns.data
    #     position_size = self.position_size
    #     if isinstance(position_size, int) or isinstance(position_size, float):
    #         weighted_returns = (1 + (base_returns * abs(position_size))) ** sign(self.position_size) - 1
    #     elif isinstance(position_size, Series):
    #         position_size = position_size.shift(1)[self.entry:self.exit]
    #         weighted_returns = (1 + (base_returns * abs(position_size))) ** sign(position_size[1]) - 1
    #     else:
    #         weighted_returns = None
    #     self._weighted_returns[self.entry:self.exit, self.ticker] = weighted_returns

    # Trade modifiers
    # TODO - revise_entry and revise_exit could be parameter setters
    def revise_entry(self, entry_day):
        '''
        revise_entry accepts an entry_day (integer), and readjusts the trade to
        start on the new (later) entry day.
        '''
        if 0 < entry_day < self.duration:
            # entry is delayed
            old_entry = self.entry
            day_before_new_entry = self.entry_event.offset(entry_day - 1)
            self.entry = self.entry_event.offset(entry_day)
            self.returns_start = self.entry_event.offset(entry_day + 1)
            self._position_loc[old_entry:day_before_new_entry, self.ticker] = 0
            return self
        else:
            return None

    def revise_exit(self, exit_day):
        '''
        revise_exit accepts an exit_day (integer), and readjusts the trade to
        end on the new exit day.
        '''
        if 0 < exit_day < self.duration:
            # trade has been shortened
            old_exit = self.exit
            self.exit = self.exit_event.offset(exit_day - self.duration)
            self.positions_end = self.exit_event.offset(exit_day - self.duration - 1)
            self._position_loc[self.exit:old_exit, self.ticker] = 0
            return self
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
        self.drawdowns().series.plot(**kwargs)

    def plot_highwater(self, **kwargs):
        self.drawdowns().highwater.plot(**kwargs)
