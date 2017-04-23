
from pandas import Series, DataFrame, qcut, cut
from numpy import sign
import matplotlib.pyplot as plt

class TradeCollection(object):

    def __init__(self, data, tickers):
        self.tickers = tickers
        self.trades = data

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trades[key]
        else:
            return [trade for trade in self.trades if trade.ticker == key]

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
            df.loc[i, cols] = filter.get(df, i)

        return df

    @property
    def count(self):
        return len(self.trades)

    @property
    def returns(self):
        return [trade.base_return for trade in self.trades]

    @property
    def durations(self):
        return [trade.duration for trade in self.trades]

    @property
    def MAEs(self):
        return [trade.MAE for trade in self.trades]

    @property
    def MFEs(self):
        return [trade.MFE for trade in self.trades]

    @property
    def mean_return(self):
        return Series(self.returns).mean()

    @property
    def std_return(self):
        return Series(self.returns).std()

    @property
    def Sharpe(self):
        returns = Series(self.returns)
        return returns.mean() / returns.std()

    @property
    def Sharpe_annual(self):
        returns = self.daily_returns()
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


    def find(self, condition):
        '''
        find accepts a lambda expression which must accept a Trade object as its input and return True/False.
        A TradeCollection of trades meeting the condition is returned.
        '''
        trades = [trade for trade in self.trades if condition(trade)]
        tickers = list(set([trade.ticker for trade in trades]))
        return TradeCollection(trades, tickers)
        

    def filter_summary(self, filter_values, bins = 5):
        '''
        filter_values is a FilterValue object. 
        Note if filter_values contains more than one type, only the first is used.
        bins is an iterable of boundary points e.g. (-1, 0, 0.5, 1, etc...), or an integer of 
        the number of bins to produce (default 5). This is passed to pandas qcut.
        '''
        trade_df = self.as_dataframe_with(filter_values)
        if isinstance(bins, int):
            type_bins = qcut(trade_df[filter_values.types[0]], bins)
        else:
            type_bins = cut(trade_df[filter_values.types[0]], bins)
        mu = trade_df.groupby(type_bins).base_return.mean()
        sd = trade_df.groupby(type_bins).base_return.std()
        N = trade_df.groupby(type_bins).base_return.count()
        return {"mean" : mu, "std" : sd, "count" : N}
        
    def filter_grouping(self, filter, bins):
        '''
        Provides a summary of filter performance for provided bins. Bins must be a sequence of boundary
        points e.g. (-1, 0, 0.25...). Each filter type will be provided as a column.
        '''
        if isinstance(bins, int):
            raise ValueError("Bins must be a sequence for filter grouping")
        trade_df = self.as_dataframe_with(filter)
        mu = DataFrame()
        sd = DataFrame()
        N = DataFrame()
        for type in filter.types:
            type_bins = cut(trade_df[type], bins)
            mu[type] = trade_df.groupby(type_bins).base_return.mean()
            sd[type] = trade_df.groupby(type_bins).base_return.std()
            N[type] = trade_df.groupby(type_bins).base_return.count()
        return {"mean" : mu, "std" : sd, "count" : N}


    def filter_comparison(self, filter1, filter2, bins1 = 5, bins2 = 5):
        '''
        Provides a matrix comparing mean, std dev, and count for each combination of filter
        values. Note only the first type of each filter is considered.
        '''
        trade_df = self.as_dataframe_with(filter1, filter2)

        f1_name = filter1.types[0]
        f2_name = filter2.types[0]
        
        if isinstance(bins1, int):
            f1_bins = qcut(trade_df[f1_name], bins1)
        else:
            f1_bins = cut(trade_df[f1_name], bins1)

        if isinstance(bins2, int):
            f2_bins = qcut(trade_df[f2_name], bins2)
        else:
            f2_bins = cut(trade_df[f2_name], bins2)

        grouping = trade_df.groupby([f1_bins, f2_bins]).base_return

        mu = DataFrame(grouping.mean()).unstack()
        sd = DataFrame(grouping.std()).unstack()
        N = DataFrame(grouping.count()).unstack()

        return {"mean" : mu, "std" : sd, "count" : N}

    def trade_frame(self, **kwargs):
        '''
        Returns a dataframe of daily cumulative return for each trade.
        Each row is a trade, and columns are days in trade.
        '''
        df = DataFrame()
        for trade in self.trades:
            df = df.append(trade.normalised, ignore_index = True)
        if df.shape[1] > 20:
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
        return (df, trade_df)

    def daily_returns(self):
        '''
        Returns an unsorted and unorderd list of daily returns for all trades.
        Used for calculating daily or annualised statistics.
        '''
        df = self.trade_frame()[0]
        daily = ((df + 1).T / (df + 1).T.shift(1)).T - 1
        returns = []
        for col in daily:
            returns.extend(daily[col].values)
        returns = Series(returns)
        returns = returns[returns.notnull()]
        return returns


    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_normalised()

    def hist(self, **kwargs):
        # First remove NaNs
        returns = [R for R in self.returns if R == R]
        plt.hist(returns, **kwargs)

    def plot_MAE(self):
        MAEs = self.MAEs
        returns = self.returns
        x_range = (min(MAEs) -0.05, 0.1)
        y_range = (min(returns) - 0.05, max(returns) + 0.05)
        plt.scatter(self.MAEs, self.returns)
        plt.plot((0, 0), y_range, color = 'black')
        plt.plot(x_range, (0, 0), color = 'black')
        plt.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        plt.xlim(x_range)
        plt.ylim(y_range)

    def plot_MFE(self):
        MFEs = self.MFEs
        returns = self.returns
        x_range = (-0.1, max(MFEs) + 0.05)
        y_range = (min(returns) - 0.05, max(returns) + 0.05)
        plt.scatter(self.MFEs, self.returns)
        plt.plot((0, 0), y_range, color = 'black')
        plt.plot(x_range, (0, 0), color = 'black')
        plt.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        plt.xlim(x_range)
        plt.ylim(y_range)

    def summary_report(self):
        '''
        Provides a summary of the trade statistics
        '''
        winners = self.find(lambda trade: trade.base_return > 0)
        losers = self.find(lambda trade: trade.base_return < 0)
        evens = self.find(lambda trade: trade.base_return == 0)
        daily_R = self.daily_returns()
        trade_df = self.as_dataframe().sort(columns = 'exit')

        win_loss = sign(trade_df.base_return).values
        previous = win_loss[0]
        win_streak = 0 + previous > 0
        biggest_win_streak = 0
        loss_streak = 0 + previous < 0
        biggest_loss_streak = 0
        for i in range(1, len(win_loss)):
            current = win_loss[i]
            if current > 0 and previous > 0:
                win_streak += 1
            elif current < 0 and previous < 0:
                loss_streak += 1
            elif current > 0 and previous < 0:
                biggest_loss_streak = max(biggest_loss_streak, loss_streak)
                loss_streak = 0
            elif current < 0 and previous > 0:
                biggest_win_streak = max(biggest_win_streak, win_streak)
                win_streak = 0
            previous = current

        trade_volume = Series(dtype = float)
        trade_volume['Number of trades'] = self.count
        trade_volume['Percent winners'] = round(100 * (winners.count / self.count), 1)
        trade_volume['Number winners'] = winners.count 
        trade_volume['Number losers'] = losers.count
        trade_volume['Number even'] = evens.count

        returns = Series(dtype = float)
        returns['Average return'] = round(100 * self.mean_return, 2)
        returns['Median return'] = round(100 * Series(self.returns).median(), 2)
        returns['Average winning return'] = round(100 * winners.mean_return, 2)
        returns['Average losing return'] = round(100 * losers.mean_return, 2)
        returns['Ratio average win to loss'] = round(winners.mean_return / abs(losers.mean_return), 2)
        returns['Largest winner'] = round(100 * max(self.returns), 2)
        returns['Largest loser'] = round(100 * min(self.returns), 2) 
        returns['Sharpe by trade'] = round(self.Sharpe, 2)
        returns['Sharpe annualised'] = round(self.Sharpe_annual, 2)
        returns['G by trade'] = round(self.G, 2)
        returns['G annualised'] = round(self.G_annual, 2)

        duration = Series(dtype = float)
        duration['Average duration'] = round(Series(self.durations).mean(), 2)
        duration['Average duration winners'] = round(Series(winners.durations).mean(), 2)
        duration['Average duration losers'] = round(Series(losers.durations).mean(), 2)
        duration['Max consecutive winners'] = biggest_win_streak
        duration['Max consecutive losers'] = biggest_loss_streak

        return (trade_volume, returns, duration)


class Trade(object):

    def __init__(self, ticker, entry_date, exit_date, entry_prices, exit_prices):
        self.ticker = ticker
        self.entry = entry_date
        self.exit = exit_date
        self.entry_price = self.get_price(entry_date, entry_prices)
        self.exit_price = self.get_price(exit_date, exit_prices)
        self.duration = (exit_date - entry_date).days
        self.normalised = Series((exit_prices[self.entry:self.exit] / self.entry_price).values) - 1
        self.cols = ["ticker", "entry", "exit", "entry_price", "exit_price", "base_return", "duration"]

    def __repr__(self):
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
        self.normalised.plot()

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

