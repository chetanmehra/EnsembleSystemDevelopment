
from pandas import Series, DataFrame, qcut, cut

class TradeCollection(object):

    def __init__(self, data, tickers):
        self.tickers = tickers
        self.trades = data

    def __getitem__(self, key):
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
    def G(self):
        # Refer research note: EQ-2011-003
        S_sqd = self.Sharpe ** 2
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
        tickers = list(set([trade.ticker for ticker in trades]))
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


    def plot_ticker(self, ticker):
        for trade in self[ticker]:
            trade.plot_normalised()

    def hist(self, **kwargs):
        # First remove NaNs
        returns = [R for R in self.returns if R == R]
        plt.hist(returns, **kwargs)



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

