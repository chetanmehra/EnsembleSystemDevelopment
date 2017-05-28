
from pandas import qcut, cut, concat, Series, DataFrame
from numpy import random
from .Metrics import GeometricGrowth, OptF

# Analysis of Filter performance with trades

class FilterPerformance():

    def __init__(self, trades):
        self.trade_df = trades.as_dataframe()
        self.result = None

    def add_filter_to_df(self, *args):
        for filter in args:
            cols = filter.types
            for col in cols:
                self.trade_df[col] = None
            if len(cols) == 1:
                cols = cols[0]
            for i in self.trade_df.index:
                self.trade_df.loc[i, cols] = filter.get(self.trade_df, i)

    def filter_summary(self, filter_values, bins = 5):
        '''
        filter_values is a FilterValue object. 
        Note if filter_values contains more than one type, only the first is used.
        bins is an iterable of boundary points e.g. (-1, 0, 0.5, 1, etc...), or an integer of 
        the number of bins to produce (default 5). This is passed to pandas qcut.
        '''
        self.add_filter_to_df(filter_values)
        if isinstance(bins, int):
            type_bins = qcut(self.trade_df[filter_values.types[0]], bins)
        else:
            type_bins = cut(self.trade_df[filter_values.types[0]], bins)
        mu = self.trade_df.groupby(type_bins).base_return.mean()
        sd = self.trade_df.groupby(type_bins).base_return.std()
        N = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}
        
    def filter_grouping(self, filter, bins):
        '''
        Provides a summary of filter performance for provided bins. Bins must be a sequence of boundary
        points e.g. (-1, 0, 0.25...). Each filter type will be provided as a column.
        '''
        if isinstance(bins, int):
            raise ValueError("Bins must be a sequence for filter grouping")
        self.add_filter_to_df(filter)
        mu = DataFrame()
        sd = DataFrame()
        N = DataFrame()
        for type in filter.types:
            type_bins = cut(self.trade_df[type], bins)
            mu[type] = self.trade_df.groupby(type_bins).base_return.mean()
            sd[type] = self.trade_df.groupby(type_bins).base_return.std()
            N[type] = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}


    def filter_comparison(self, filter1, filter2, bins1 = 5, bins2 = 5):
        '''
        Provides a matrix comparing mean, std dev, and count for each combination of filter
        values. Note only the first type of each filter is considered.
        '''
        self.add_filter_to_df(filter1, filter2)

        f1_name = filter1.types[0]
        f2_name = filter2.types[0]
        
        if isinstance(bins1, int):
            f1_bins = qcut(self.trade_df[f1_name], bins1)
        else:
            f1_bins = cut(self.trade_df[f1_name], bins1)

        if isinstance(bins2, int):
            f2_bins = qcut(self.trade_df[f2_name], bins2)
        else:
            f2_bins = cut(self.trade_df[f2_name], bins2)

        grouping = self.trade_df.groupby([f1_bins, f2_bins]).base_return

        mu = DataFrame(grouping.mean()).unstack()
        sd = DataFrame(grouping.std()).unstack()
        N = DataFrame(grouping.count()).unstack()

        self.result = {"mean" : mu, "std" : sd, "count" : N}

    def plot_Sharpe(self):
        Sharpe = self.result['mean'] / self.result['std']
        Sharpe.plot()

    def plot(self):
        mean_plus = self.result['mean'] + self.result['std']
        mean_minus = self.result['mean'] - self.result['std']
        self.result['mean'].plot()
        mean_plus.plot(style = '-')
        mean_minus.plot(style = '-')



# Trade Summary Report

def summary_trade_volume(trades):
    winners = trades.find(lambda trade: trade.base_return > 0)
    losers = trades.find(lambda trade: trade.base_return < 0)
    evens = trades.find(lambda trade: trade.base_return == 0)
        
    trade_volume = Series(dtype = float)
    trade_volume['Number of trades'] = trades.count
    trade_volume['Percent winners'] = round(100 * (float(winners.count) / trades.count), 1)
    trade_volume['Number winners'] = winners.count 
    trade_volume['Number losers'] = losers.count
    trade_volume['Number even'] = evens.count
    return trade_volume

def summary_returns(trades):
    winners = trades.find(lambda trade: trade.base_return > 0)
    losers = trades.find(lambda trade: trade.base_return < 0)
    evens = trades.find(lambda trade: trade.base_return == 0)

    returns = Series(dtype = float)
    returns['Average return'] = round(100 * trades.mean_return, 2)
    returns['Average return inc slippage'] = round(100 * trades.returns_slippage.mean(), 2)
    returns['Median return'] = round(100 * trades.returns.median(), 2)
    returns['Average winning return'] = round(100 * winners.mean_return, 2)
    returns['Average losing return'] = round(100 * losers.mean_return, 2)
    returns['Ratio average win to loss'] = round(winners.mean_return / abs(losers.mean_return), 2)
    returns['Largest winner'] = round(100 * max(trades.returns), 2)
    returns['Largest loser'] = round(100 * min(trades.returns), 2) 
    returns['Sharpe by trade'] = round(trades.Sharpe, 2)
    returns['Sharpe by trade inc slippage'] = round(trades.returns_slippage.mean() / trades.returns_slippage.std(), 2)
    returns['Sharpe annualised'] = round(trades.Sharpe_annual, 2)
    returns['Sharpe annualised inc slippage'] = round(trades.Sharpe_annual_slippage, 2)
    returns['Opt F'] = round(OptF(trades.returns), 2)
    returns['G by trade'] = round(trades.G, 2)
    returns['G annualised'] = round(trades.G_annual, 2)
    # HACK geometric growth rate has assumed in the market 100%
    returns['Geom. growth rate'] = round(GeometricGrowth(trades.returns, 250 / trades.durations.mean()), 2)
    return returns

def summary_duration(trades):
    winners = trades.find(lambda trade: trade.base_return > 0)
    losers = trades.find(lambda trade: trade.base_return < 0)
    evens = trades.find(lambda trade: trade.base_return == 0)
    positive_runs, negative_runs = trades.consecutive_wins_losses()

    duration = Series(dtype = float)
    duration['Average duration'] = round(trades.durations.mean(), 2)
    duration['Average duration winners'] = round(winners.durations.mean(), 2)
    duration['Average duration losers'] = round(losers.durations.mean(), 2)
    duration['Max consecutive winners'] = positive_runs.max()
    duration['Max consecutive losers'] = negative_runs.max()
    duration['Avg consecutive winners'] = round(positive_runs.mean(), 2)
    duration['Avg consecutive losers'] = round(negative_runs.mean(), 2)
    return duration

def summary_report(trades):
    '''
    Provides a summary of the trade statistics
    '''
    trade_volume = summary_trade_volume(trades)
    returns = summary_returns(trades)
    duration = summary_duration(trades)
    return concat((trade_volume, returns, duration))

def summary_by_period(trades, periods = 5):
    entries = [T.entry for T in trades.as_list()]
    entries.sort()
    bin_size = round(trades.count / periods) + 1
    bin_size = max(bin_size, 25)
    start, end = (0, bin_size)
    period_summary = DataFrame(dtype = float)
    while start < trades.count:
        end = min(end, trades.count)
        start_date = entries[start]
        end_date = entries[end - 1]
        label = '{}:{}'.format(start_date.strftime('%b-%y'), end_date.strftime('%b-%y'))
        subset = trades.find(lambda trade: start_date <= trade.entry <= end_date)
        period_summary[label] = summary_report(subset)
        start = end + 1
        end += bin_size
    return period_summary



# Market stability
# Performs cross-validation on several sub sets of the market

def cross_validate_trades(trades, N = 20, subset_fraction = 0.7):
    
    tickers = trades.tickers
    sample_size = round(len(tickers) * subset_fraction)
    summary = DataFrame(dtype = float)

    for n in range(N):
        sample_tickers = list(random.choice(tickers, sample_size, replace = False))
        trade_subset = trades.find(lambda T: T.ticker in sample_tickers)
        summary[n] = summary_report(trade_subset)

    result = DataFrame(dtype = float)
    result['Base'] = summary_report(trades)
    result['Mean'] = summary.mean(axis = 1)
    result['Std'] = summary.std(axis = 1)
    result['Median'] = summary.median(axis = 1)
    result['Max'] = summary.max(axis = 1)
    result['Min'] = summary.min(axis = 1)

    return (result, summary)


def cross_validate_positions(strategy, positionSelector, N = 20, subset_fraction = 0.7):

    trades = strategy.trades
    original_positions = strategy.positions
    tickers = trades.tickers
    start_date = strategy.positions.start
    
    strategy.positions = positionSelector(strategy)
    base_returns = strategy.returns
    
    sample_size = round(len(tickers) * subset_fraction)

    for n in range(N):
        sample_tickers = list(random.choice(tickers, sample_size, replace = False))
        trade_subset = trades.find(lambda T: T.ticker in sample_tickers)
        strategy.trades = trade_subset    
        strategy.positions = positionSelector(strategy)
        sub_returns = strategy.returns.plot(start = start_date, color = 'grey')

    base_returns.plot(start = start_date, color = 'black')
    strategy.market_returns.plot(start = start_date, color = 'red')
    strategy.positions = original_positions


