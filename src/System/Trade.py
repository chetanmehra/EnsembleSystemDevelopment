
from pandas import Series, DataFrame, qcut, cut, concat
from numpy import sign, log, hstack
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class TradeCollection(object):

    def __init__(self, data, tickers):
        self.tickers = tickers
        self.trades = data
        self.slippage = 0.011
        self._returns = None
        self._durations = None
        self._daily_returns = None

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
        plt.scatter(MFEs, returns)
        plt.plot((0, 0), y_range, color = 'black')
        plt.plot(x_range, (0, 0), color = 'black')
        plt.plot((x_range[0], y_range[1]), (x_range[0], y_range[1]), color = 'red')
        plt.xlim(x_range)
        plt.ylim(y_range)


    def before_after(self, tf, i):
        '''
        tf is expected to be a dataframe of daily returns.
        Returns data for trend calculation
        '''
        tf = log(tf + 1)
        X = tf.iloc[:, :(i + 1)].sum(axis = 1)
        Y = tf.iloc[:, (i + 1):].sum(axis = 1)
        nulls = (X * Y).isnull()
        X = X[~nulls]
        X = X.reshape(X.count(), 1)
        Y = Y[~nulls]
        return (X, Y)

    def plot_trends2(self, n_trends = 8):
        lm = LinearRegression()
        color_map = plt.get_cmap('jet')
        color_index = 80
        color_inc = round(120 / n_trends)
        tf = self.trade_frame(compacted = False, cumulative = False)
        for i in range(1, n_trends):
            column = tf.columns[i]
            X, R = self.before_after(tf, i)
            lm.fit(X, R)
            plt.plot(X, lm.predict(X), color = color_map(color_index), label = str(column))
            color_index += color_inc
        plt.plot(plt.xlim(), (0, 0), color = 'black')
        plt.plot((0, 0), plt.ylim(), color = 'black')
        plt.legend(loc = "upper left", fontsize = 8)


    def plot_trends(self, n_trends = 5):
        lm = LinearRegression()
        color_map = plt.get_cmap('jet')
        color_index = 80
        color_inc = round(120 / n_trends)
        tf = self.trade_frame(compacted = False)
        tf['R'] = self.returns
        for i in range(0, n_trends):
            column = tf.columns[i]
            X = tf[column]
            nulls = (X * tf['R']).isnull()
            X = X[~nulls]
            X = X.reshape(X.count(), 1)
            R = tf['R'][~nulls]
            lm.fit(X, R)
            plt.plot(X, lm.predict(X), color = color_map(color_index), label = str(column))
            color_index += color_inc
        plt.plot(plt.xlim(), (0, 0), color = 'black')
        plt.plot((0, 0), plt.ylim(), color = 'black')
        plt.legend(loc = "upper left", fontsize = 8)


    def summary_trade_volume(self):
        winners = self.find(lambda trade: trade.base_return > 0)
        losers = self.find(lambda trade: trade.base_return < 0)
        evens = self.find(lambda trade: trade.base_return == 0)
        
        trade_volume = Series(dtype = float)
        trade_volume['Number of trades'] = self.count
        trade_volume['Percent winners'] = round(100 * (float(winners.count) / self.count), 1)
        trade_volume['Number winners'] = winners.count 
        trade_volume['Number losers'] = losers.count
        trade_volume['Number even'] = evens.count
        return trade_volume

    def summary_returns(self):
        winners = self.find(lambda trade: trade.base_return > 0)
        losers = self.find(lambda trade: trade.base_return < 0)
        evens = self.find(lambda trade: trade.base_return == 0)

        returns = Series(dtype = float)
        returns['Average return'] = round(100 * self.mean_return, 2)
        returns['Average return inc slippage'] = round(100 * self.returns_slippage.mean(), 2)
        returns['Median return'] = round(100 * self.returns.median(), 2)
        returns['Average winning return'] = round(100 * winners.mean_return, 2)
        returns['Average losing return'] = round(100 * losers.mean_return, 2)
        returns['Ratio average win to loss'] = round(winners.mean_return / abs(losers.mean_return), 2)
        returns['Largest winner'] = round(100 * max(self.returns), 2)
        returns['Largest loser'] = round(100 * min(self.returns), 2) 
        returns['Sharpe by trade'] = round(self.Sharpe, 2)
        returns['Sharpe by trade inc slippage'] = round(self.returns_slippage.mean() / self.returns_slippage.std(), 2)
        returns['Sharpe annualised'] = round(self.Sharpe_annual, 2)
        returns['Sharpe annualised inc slippage'] = round(self.Sharpe_annual_slippage, 2)
        returns['G by trade'] = round(self.G, 2)
        returns['G annualised'] = round(self.G_annual, 2)
        return returns

    def summary_duration(self):
        winners = self.find(lambda trade: trade.base_return > 0)
        losers = self.find(lambda trade: trade.base_return < 0)
        evens = self.find(lambda trade: trade.base_return == 0)
        positive_runs, negative_runs = self.consecutive_wins_losses()

        duration = Series(dtype = float)
        duration['Average duration'] = round(self.durations.mean(), 2)
        duration['Average duration winners'] = round(winners.durations.mean(), 2)
        duration['Average duration losers'] = round(losers.durations.mean(), 2)
        duration['Max consecutive winners'] = positive_runs.max()
        duration['Max consecutive losers'] = negative_runs.max()
        duration['Avg consecutive winners'] = round(positive_runs.mean(), 2)
        duration['Avg consecutive losers'] = round(negative_runs.mean(), 2)
        return duration

    def summary_report(self):
        '''
        Provides a summary of the trade statistics
        '''
        trade_volume = self.summary_trade_volume()
        returns = self.summary_returns()
        duration = self.summary_duration()
        return concat((trade_volume, returns, duration))


class Trade(object):

    def __init__(self, ticker, entry_date, exit_date, entry_prices, exit_prices):
        self.ticker = ticker
        self.entry = entry_date
        self.exit = exit_date
        self.entry_price = self.get_price(entry_date, entry_prices)
        self.exit_price = self.get_price(exit_date, exit_prices)
        self.normalised = Series((exit_prices[self.entry:self.exit] / self.entry_price).values) - 1
        # Note: duration is measured by days-in-market not calendar days
        self.duration = len(self.normalised)
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

