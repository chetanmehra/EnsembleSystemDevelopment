# System analysis

import matplotlib.pyplot as plt
from pandas import qcut, cut, concat, DataFrame, Series
from numpy import log, random, arange, NaN
from multiprocessing import Pool, cpu_count

from data_types.constants import TRADING_DAYS_PER_YEAR
from data_types.returns import Returns, AverageReturns
from system.core import Strategy, Portfolio
from system.metrics import *
from trade_modifiers.exit_conditions import StopLoss, TrailingStop


# Path dependent trade results
# Attempts to characterise trade outcomes based on path taken.

# Mean and std deviation
# Winners vs losers
def win_loss_trace(trades):

    tf = trades.trade_frame(compacted = False, cumulative = True)
    rets = trades.returns

    winners = tf[rets > 0]
    losers = tf[rets <= 0]

    win_N = winners.count()
    win_cutoff = max(win_N[win_N > max(round(win_N[0] * 0.3), 20)].index)
    los_N = losers.count()
    los_cutoff = max(los_N[los_N > max(round(los_N[0] * 0.3), 20)].index)
    cutoff = min(win_cutoff, los_cutoff)

    win_mean = winners.mean()[:cutoff]
    win_std = winners.std()[:cutoff]
    los_mean = losers.mean()[:cutoff]
    los_std = losers.std()[:cutoff]

    for trace in [('blue', win_mean, win_std), ('red', los_mean, los_std)]:
        trace[1].plot(color = trace[0])
        (trace[1] + trace[2]).plot(style = '--', color = trace[0])
        (trace[1] - trace[2]).plot(style = '--', color = trace[0])


def std_dev_trace(trades):

    tf = trades.trade_frame(compacted = False, cumulative = True)
    rets = trades.returns

    tf_N = tf.count()
    cutoff = max(tf_N[tf_N > max(round(tf_N[0] * 0.3), 20)].index)

    tf_normalised = tf.loc[:, :cutoff]
    tf_normalised = (tf_normalised - tf_normalised.mean()) / tf_normalised.std()


    winners = tf_normalised[rets > 0]
    losers = tf_normalised[rets <= 0]

    win_mean = winners.mean()
    los_mean = losers.mean()

    for trace in [('blue', win_mean), ('red', los_mean)]:
        trace[1].plot(color = trace[0])



def result_after_delay(trades, N):
    # Calculate the remaining returns if entry is lagged.
    # This will establish if short duration losers are impacting the results.
    tf = trades.trade_frame(compacted = False, cumulative = False)
    cutoff = min(tf.shape[1], N)
    tf_log = log(tf + 1)
    result = DataFrame(columns = list(range(cutoff)), dtype = float)
    for i in range(cutoff):
        result[i] = tf_log.loc[:, i:].sum(axis = 1)
    return result


# Boxplot by return outcomes
def box_colour(bp, box_num, edge_color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color = edge_color)

def boxplot_path_by_outcome(trades, day):
    tf = trades.trade_frame(compacted = False, cumulative = False)
    # Get the daily returns from the day after the requested day onwards.
    # Remove any trades which are empty moving forward, as we know these would have been closed.
    forward = tf.loc[:, (day + 1):].dropna(how = 'all')
    forward = log(forward + 1)
    backward = tf.loc[forward.index, :day]
    backward = log(backward + 1)

    df = DataFrame(dtype = float)
    df['Final Return'] = qcut(forward.sum(axis = 1).round(2), 5)
    df['Current Return'] = backward.sum(axis = 1)

    bp = df.boxplot('Current Return', by = 'Final Return', return_type = 'dict')

    whisker_points = []
    [whisker_points.extend(list(whisker.get_ydata())) for whisker in bp[0]['whiskers']]
    y_min = min(whisker_points) * 1.1
    y_max = max(whisker_points) * 1.1
    plt.ylim((y_min, y_max))
    plt.xticks(fontsize = 'small', rotation = 30)
    plt.ylabel('Current Return')
    plt.title('Day {}'.format(day))

# TODO implement boxplot_outcome_by_path
def boxplot_outcome_by_path(trades, day):
    pass


# Plot trends of trade collection
def prep_for_lm(x, y):
    nulls = (x * y).isnull()
    x = x[~nulls]
    x = x.reshape(x.count(), 1)
    y = y[~nulls]
    return (x, y)

def tf_sum(tf, start, end):
    tf_log = log(tf + 1)
    X = tf_log.iloc[:, start:end].sum(axis = 1)
    return X

def tf_sharpe(tf, start, end):
    X_mean = tf.iloc[:, start:end].mean(axis = 1)
    X_std = tf.iloc[:, start:end].std(axis = 1)
    return X_mean / X_std


def calc_trends(trades, dep_time, dep_method, indep_method):
    allowable_timing = ['end', 'interim']
    allowable_dep_methods = {
        'sum' : tf_sum,
        'sharpe' : tf_sharpe
        }
    allowable_indep_methods = {
        'sum' : tf_sum
        }
    if dep_time not in allowable_timing:
        raise ValueError('dep_timing must be one of: {}'.format(', '.join(allowable_timing)))
    try:
        dep_method = allowable_dep_methods[dep_method]
    except KeyError:
        raise ValueError('dep_method must be one of: {}'.format(', '.join(allowable_dep_methods.keys())))
    try:
        indep_method = allowable_indep_methods[indep_method]
    except KeyError:
        raise ValueError('indep_method must be one of: {}'.format(', '.join(allowable_indep_methods.keys())))

    lm = LinearRegression()
    tf = trades.trade_frame(compacted = False, cumulative = False)
    trends = DataFrame([-0.1, -0.01, 0, 0.01, 0.1])
    result = DataFrame(None, None, trends[0], dtype = float)

    if dep_time == 'end':
        R = dep_method(tf, None, None)
        
    i = j = 1
    while j < tf.shape[1] and tf[tf.columns[j]].count() >= 20:
        X = indep_method(tf, None, (j + 1))
        if dep_time == 'interim':
            R = dep_method(tf, (j + 1), None)
        X, R_i = prep_for_lm(X, R)
        lm.fit(X, R_i)
        result.loc[j, :] = lm.predict(trends)
        k = j
        j += i
        i = k
    return result


def plot_trends(trades):
    f, axarr = plt.subplots(2, 2, sharex = True)
    sp00 = calc_trends(trades, dep_time = 'interim', dep_method = 'sum', indep_method = 'sum')
    sp10 = calc_trends(trades, dep_time = 'interim', dep_method = 'sharpe', indep_method = 'sum')
    sp01 = calc_trends(trades, dep_time = 'end', dep_method = 'sum', indep_method = 'sum')
    sp11 = calc_trends(trades, dep_time = 'end', dep_method = 'sharpe', indep_method = 'sum')

    sp00.plot(ax = axarr[0, 0])
    sp10.plot(ax = axarr[1, 0])
    sp01.plot(ax = axarr[0, 1])
    sp11.plot(ax = axarr[1, 1])
        
    axarr[0, 0].set_xscale('log')
    axarr[1, 0].set_xscale('log')
    axarr[0, 1].set_xscale('log')
    axarr[1, 1].set_xscale('log')

    axarr[0, 0].set_title('Interim')
    axarr[0, 0].set_ylabel('Sum')
    axarr[1, 0].set_ylabel('Sharpe')
    axarr[0, 1].set_title('End')



# TODO Test signal performance against TrendBenchmark
class TrendBenchmark(object):
    '''
    The TrendBenchmark is not intended for use in a strategy, but for testing the performance
    of an indicator against ideal, perfect hindsight identification of trends.
    '''
    def __init__(self, period):
        self.period = period
        
    def __call__(self, strategy):
        prices = strategy.get_indicator_prices()
        trend = DataFrame(None, index = prices.index, columns = prices.columns, dtype = float)
        last_SP = Series(None, index = prices.columns)
        current_trend = Series('-', index = prices.columns)
        for i in range(prices.shape[0] - self.period):
            # If there are not any new highs in the recent period then must have been 
            # a swing point high.
            SPH = ~(prices.iloc[(i + 1):(i + self.period)] > prices.iloc[i]).any()
            # NaN in series will produce false signals and need to be removed
            SPH = SPH[prices.iloc[i].notnull()]
            SPH = SPH[SPH]
            # Only mark as swing point high if currently in uptrend or unidentified trend, otherwise ignore.
            SPH = SPH[current_trend[SPH.index] != 'DOWN']
            if not SPH.empty:
                current_trend[SPH.index] = 'DOWN'
                trend.loc[trend.index[i], SPH.index] = prices.iloc[i][SPH.index]
            # Repeat for swing point lows.
            SPL = ~(prices.iloc[(i + 1):(i + self.period)] < prices.iloc[i]).any()
            SPL = SPL[prices.iloc[i].notnull()]
            SPL = SPL[SPL]
            SPL = SPL[current_trend[SPL.index] != 'UP']
            if not SPL.empty:
                current_trend[SPL.index] = 'UP'
                trend.loc[trend.index[i], SPL.index] = prices.iloc[i][SPL.index]
        self.trend = trend.interpolate()


# Analysis of Filter performance with trades
class FilterPerformance():

    def __init__(self, trades):
        self.trades = trades
        self.trade_df = trades.as_dataframe()
        self.result = None

    def add(self, *args):
        for filter in args:
            self.trade_df = self.trades.add_to_df(self.trade_df, filter)

    def summarise(self, filter_values, bins = 5):
        '''
        filter_values is the name of one of the filters already added to the 
        trade_df. 
        bins is an iterable of boundary points e.g. (-1, 0, 0.5, 1, etc...), or an integer of 
        the number of bins to produce (default 5). This is passed to pandas qcut.
        '''
        try:
            values = self.trade_df[filter_values]
        except KeyError as E:
            print("{0} not yet assessed. Please add with 'add_to_df' and try again.".format(filter_values))
            return None

        if isinstance(bins, int):
            type_bins = qcut(self.trade_df[filter_values], bins)
        else:
            type_bins = cut(self.trade_df[filter_values], bins)
        mu = self.trade_df.groupby(type_bins).base_return.mean()
        sd = self.trade_df.groupby(type_bins).base_return.std()
        N = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}
        return self.result
        
    def group(self, filter, bins):
        '''
        Provides a summary of filter performance for provided bins. Bins must be a sequence of boundary
        points e.g. (-1, 0, 0.25...). Each filter type will be provided as a column.
        '''
        if isinstance(bins, int):
            raise ValueError("Bins must be a sequence for filter grouping")
        self.add(filter)
        mu = DataFrame()
        sd = DataFrame()
        N = DataFrame()
        for type in filter.types:
            type_bins = cut(self.trade_df[type], bins)
            mu[type] = self.trade_df.groupby(type_bins).base_return.mean()
            sd[type] = self.trade_df.groupby(type_bins).base_return.std()
            N[type] = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}

    def compare(self, f1_name, f2_name, bins1 = 5, bins2 = 5):
        '''
        Provides a matrix comparing mean, std dev, and count for each combination of filter
        values. Note only the first type of each filter is considered.
        '''
        
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

    def trailing_count(self):
        removed = self.result['count'].shift(1)
        removed.iloc[0] = 0
        return self.result['count'].sum() - removed.cumsum()

    def set_x_labels(self, ax):
        x_labels = self.result['count'].index
        ax.set_xticks(arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        plt.setp(ax.get_xticklabels(), 
                        rotation=45, 
                        ha="right", 
                        rotation_mode="anchor", 
                        fontsize = 8)

    def plot_count(self, ax, **kwargs):
        count = self.trailing_count()
        ax2 = ax.twinx()
        count.plot(ax = ax2, **kwargs)
        ax2.set_ylabel('Count')
        self.set_x_labels(ax2)

    def plot_Sharpe(self, **kwargs):
        Sharpe = self.result['mean'] / self.result['std']
        if 'title' not in kwargs:
            kwargs['title'] = self.result['mean'].index.name + ' Sharpe'
        ax = Sharpe.plot(**kwargs)
        ax.set_ylabel('Sharpe ratio')

    def plot_mean(self, **kwargs):
        mean_plus = self.result['mean'] + self.result['std']
        mean_minus = self.result['mean'] - self.result['std']
        ax = self.result['mean'].plot(**kwargs)
        kwargs['style'] = '-'
        mean_plus.plot(**kwargs)
        mean_minus.plot(**kwargs)
        ax.set_ylabel('Mean')
        return ax


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
    #returns['Sharpe by trade'] = round(trades.Sharpe, 2)
    #returns['Sharpe by trade inc slippage'] = round(trades.returns_slippage.mean() / trades.returns_slippage.std(), 2)
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
    # Whilst consecutive winners/losers is interesting I'm not confident
    # that this is being calculated by the TradeCollection in a meaningful way.
    # duration['Max consecutive winners'] = positive_runs.max()
    # duration['Max consecutive losers'] = negative_runs.max()
    # duration['Avg consecutive winners'] = round(positive_runs.mean(), 2)
    # duration['Avg consecutive losers'] = round(negative_runs.mean(), 2)
    return duration

def summary_report(trades = None, **kwargs):
    '''
    Provides a summary of the trade statistics
    '''
    if trades is not None:
        trade_volume = summary_trade_volume(trades)
        returns = summary_returns(trades)
        duration = summary_duration(trades)
        return concat((trade_volume, returns, duration))
    else:
        df = DataFrame()
        for label, trades in kwargs.items():
            df[label] = summary_report(trades)
        return df


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



class Sampler:

    def __init__(self, N = 20, subset_fraction = 0.7):
        self.N = N
        self.subset_fraction = subset_fraction
        self.clear()

    def clear(self):
        self.sampled_returns = DataFrame(dtype = float)
        self.summary_report = DataFrame(dtype = float)

    def check_robustness(self, returns):
        '''
        check_robustness takes a Returns object and samples at random
        according to the Sampler parameters. The sampled returns are stored
        for later processing in the `sampled_returns` attribute.
        '''
        tickers = returns.columns
        ReturnsClass = returns.__class__
        sample_size = round(len(tickers) * self.subset_fraction)
        self.sampled_returns['selected'] = returns.combined().data
        print('Running {} samples: '.format(self.N), end = '')
        for n in range(self.N):
            sample_tickers = list(random.choice(tickers, sample_size, replace = False))
            subset = ReturnsClass(returns[sample_tickers])
            self.sampled_returns[n] = subset.combined().data
            print('.', end = '')
        self.sampled_returns = AverageReturns(self.sampled_returns)
        print('\ndone.')

    def check_selection_skill(self, selected, base):
        '''
        check_select_skill attempts to assess the skill in a selection process
        (e.g. a filter) vs a random selection of the same size from the base
        collection.
        selected - the TradeCollection produced by the selection process
        base - the TradeCollection representing the pool of trades selected from
        '''
        self.summary_report['selected'] = summary_report(selected)
        print('Running {} samples: '.format(self.N), end = '')
        for n in range(self.N):
            samples = random.choice(range(base.count), selected.count, replace = False)
            # random.choice returns an array of numpy.int32, so we need to cast to int
            samples = [int(s) for s in samples]
            trade_subset = base.subset(samples)
            self.summary_report[n] = summary_report(trade_subset)
            print('.', end = '')
        print('\ndone.')

    def sample_hist(self, bins = 20):
        '''
        Using the summary dataframe from a robustness check and plots
        the mean, std dev, and Sharpe.
        '''
        keys = ['mean', 'std. dev.', 'Sharpe']
        mean = self.sampled_returns.annual_mean()
        std = self.sampled_returns.volatility()
        sharpe = self.sampled_returns.sharpe()
        results = [mean, std, sharpe]
        fig, axarr = plt.subplots(1, 3, sharey=True)
        for i in range(3):
            results[i][1:].hist(ax = axarr[i], bins = bins)
            axarr[i].set_title(keys[i])
            axarr[i].axvline(results[i]['selected'], color='k', linestyle='dashed', linewidth=1)    
        return (fig, axarr)

    def skill_hist(self, metric, **kwargs):
        values = self.summary_report.loc[metric]
        # first value is the selected, so we remove from the histogram
        ax = values[1:].hist(**kwargs)
        ax.axvline(values['selected'], color='k', linestyle='dashed', linewidth=1)
        return ax


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


def cross_validate_portfolio(portfolio, N = 20, subset_fraction = 0.7):
    tickers = portfolio.strategy.market.tickers
    sample_size = round(len(tickers) * subset_fraction)
    summary = {
        'market' : DataFrame(dtype = float), 
        'portfolio' : DataFrame(dtype = float)
    }

    for n in range(N):
        sample_tickers = list(random.choice(tickers, sample_size, replace = False))
        strat_subset = portfolio.strategy.subset(sample_tickers)
        sub_portfolio = Portfolio(strat_subset, portfolio.starting_capital)
        sub_portfolio.run()
        summary['market'][n] = strat_subset.market_returns.combined().data
        summary['portfolio'][n] = sub_portfolio.returns.data
    
    summary['market']['base'] = portfolio.strategy.market_returns.combined().data
    summary['portfolio']['base'] = portfolio.returns.data
    summary['market'] = Returns(summary['market'])
    summary['portfolio'] = Returns(summary['portfolio'])
    
    return summary


def cross_validate_positions(strategy, N = 20, subset_fraction = 0.7):

    trades = strategy.trades
    original_positions = strategy.positions.copy()
    tickers = trades.tickers
    start_date = strategy.positions.start
    
    base_returns = strategy.returns
    
    sample_size = round(len(tickers) * subset_fraction)

    for n in range(N):
        sample_tickers = list(random.choice(tickers, sample_size, replace = False))
        trade_subset = trades.find(lambda T: T.ticker in sample_tickers)
        strategy.trades = trade_subset    
        strategy.positions.update_from_trades(trade_subset)
        sub_returns = strategy.returns.plot(start = start_date, color = 'grey')

    base_returns.plot(start = start_date, color = 'black')
    strategy.market_returns.plot(start = start_date, color = 'red')
    strategy.positions = original_positions
    strategy.trades = trades


class ParameterFuzzer:

    def __init__(self, strategy, base_parameters, processes = None):
        self.strategy = strategy
        if strategy.trades is None:
            strategy.run()
        self.base = summary_report(strategy.trades)
        self.base_pars = base_parameters
        if processes is None:
            processes = cpu_count()
        self.processes = processes
        self._fuzzed_pars = None
        self.results = None
        self.summary = None

    @property
    def metrics(self):
        return self.base.index

    @property
    def parameter_tuples(self):
        # Note: below assumes last column in fuzzed_pars is 'Fuzz size'
        return [tuple(pars) for pars in self.fuzzed_pars[self.fuzzed_pars.columns[:-1]].values]

    @property
    def fuzzed_strategies(self):
        output = []
        for pars in self.parameter_tuples:
            strategy = self.strategy.copy()
            strategy.signal_generator.update_param(pars)
            output.append((pars, strategy))
        return output

    @property
    def fuzzed_pars(self):
        return self._fuzzed_pars

    @fuzzed_pars.setter
    def fuzzed_pars(self, values):
        if isinstance(values, DataFrame):
            if 'Fuzz size' not in values.columns:
                # We assume the base parameters are the first row
                values['Fuzz size'] = self.fuzz_size(values)
            self._fuzzed_pars = values
        elif isinstance(values, list):
            # We assume we have a list of tuples
            values = DataFrame(values)
            values['Fuzz size'] = self.fuzz_size(values)
            self._fuzzed_pars = values
        else:
            raise TypeError("values not recognised, must be DataFrame or list of tuples")

    def fuzz(self):
        '''
        fuzz runs the strategy with the fuzzed_pars and collates the results
        '''
        pool = Pool(processes = self.processes)
        self.results = pool.map(self.strategy_runner, self.fuzzed_strategies)


    def strategy_runner(self, inputs):
        '''
        strategy_runner is designed to be run in parallel processing.
        It runs the strategy with the provided parameters
        inputs - a tuple (parameters, strategy)
        '''
        pars = inputs[0]
        strategy = inputs[1]
        try:
            strategy.run()
        except:
            pass
        return (pars, strategy)
        
    def summarise(self):
        if self.results is None:
            self.fuzz()
        pool = Pool(processes = self.processes)
        summaries = pool.map(self.summary_runner, self.results)
        summary_table = DataFrame(dict(summaries))
        # TODO there's no guarantee that the results are in the same order as fuzz size
        summary_table.loc['Fuzz size'] = self.fuzzed_pars['Fuzz size'].values
        self.summary = summary_table

    def summary_runner(self, result):
        '''
        summary_runner is designed to support parallel processing
        It takes the strategy results and calculates the summary report 
        outputting a tuple of (label, summary_report).
        The result input should contain the parameter tuple, and strategy.
        '''
        # result[0] will be the parameter tuple
        # result[1] will be the strategy
        label = ':'.join(['{:0.1f}'] * len(result[0])).format(*result[0])
        return (label, summary_report(result[1].trades))

    def fuzz_parameters(self, fuzz_range = 0.15, num_trials = 20):
        '''
        fuzz_parameters takes a base set of parameters and creates random variations
        fuzz_range - the (approx) percent variation (+/-) to apply to the parameters
        num_trials - the number of samples to generate
        '''
        parameter_labels = ['P' + str(n) for n in list(range(len(self.base_pars)))]

        multiples = DataFrame((1 - fuzz_range) + random.rand(num_trials, len(self.base_pars)) * (2 * fuzz_range), columns = parameter_labels)
        fuzzed_pars = DataFrame(0, index = list(range(num_trials)), columns = parameter_labels)

        for par in parameter_labels:
            fuzzed_pars[par] = self.base_pars[parameter_labels.index(par)] * multiples[par]

        fuzzed_pars['Fuzz size'] = self.fuzz_size(fuzzed_pars, multiples)
        self.fuzzed_pars = fuzzed_pars

    def fuzz_size(self, fuzzed_pars, multiples = None):
        '''
        fuzz_size takes a dataframe of multiples (i.e. variation * base parameters)
        and produces a metric of how far the sample is from the base parameter set.
        If multiples are not provided they will be calculated from the provided fuzzed_pars
        '''
        if multiples is None:
            multiples = fuzzed_pars / self.base_pars
        fuzz_size = ((multiples - 1) ** 2).sum(axis = 'columns') ** 0.5
        fuzz_magnitude = (fuzzed_pars ** 2).sum(axis = 'columns') - (Series(self.base_pars) ** 2).sum()
        return (fuzz_size * sign(fuzz_magnitude)).round(2)

    def summarise_metric(self, metric):
        columns = self.summary.columns
        # assumes headings are of form '##.#:##.#' e.g. '100.0:50.0'
        df = DataFrame()
        values = self.summary.loc[metric]
        for c in columns:
            row = int(c.split(".")[0])
            col = float(c.split(":")[1]) / row
            df.loc[row, col] = values[c]
        return df

    def plot_metric(self, metric, **kwargs):
        df = self.summarise_metric(metric)
        if 'title' not in kwargs:
            kwargs['title'] = metric
        ax = df.plot(**kwargs)
        ax.legend()
        ax.set_title(metric)
        return ax


    def plot(self, metric):
        plt.scatter(self.summary.loc['Fuzz size'], self.summary.loc[metric])
        plt.scatter(0, self.base.loc[metric], color = 'red')


    def signal_coincidence(self, strategy):
        '''
        Assesses how often entry signals arrive and typical number
        arriving together. The thinking being that you want signals to arrive 
        often, but separately.
        Returns a tuple of ('% of days with entries', 'Average no. of coincident entries')
        '''
        event_df = strategy.events.as_dataframe()
        events_per_date = event_df.date.value_counts()
        percent_events = len(events_per_date) / len(strategy.market.index)
        average_coincidence = events_per_date.mean()
        return (percent_events, average_coincidence)


# Analysis of exit conditions
def test_trailing_stop(strat, stops):
    result = DataFrame(columns = [0] + stops)
    result[0] = summary_report(strat.trades)
    for s in stops:
        result[s] = summary_report(strat.trades.apply_exit_condition(TrailingStop(s)))
    return result

def test_stop_loss(strat, stops):
    result = DataFrame(columns = [0] + stops)
    result[0] = summary_report(strat.trades)
    for s in stops:
        result[s] = summary_report(strat.trades.apply_exit_condition(StopLoss(s)))
    return result


class PortfolioAnalyser:
    """
    The PortfolioAnalyser as its name suggests analyses the performance of a Portfolio.
    In particular it provides methods for assessing the robustness of the Portfolio 
    settings in terms of target number of positions, or starting capital.
    """
    def __init__(self, portfolio):
        if portfolio.trades is None:
            portfolio.run_events()
        self.portfolio = portfolio
        self.subsets = {}

    def clear(self):
        self.subsets = {}

    def vary_starting_capital(self, variations = [10000, 15000, 20000, 25000, 30000]):
        for capital in variations:
            variation = self.portfolio.copy()
            variation.change_starting_capital(capital)
            variation.run_events()
            self.subsets["Starting_Cash_" + str(round(capital / 1000, 0)) + "k"] = variation

    def vary_target_positions(self, variations = [3, 4, 5, 6, 7, 8, 9, 10]):
        var = 1
        for positions in variations:
            print("Running variation", var, "of", len(variations))
            var += 1
            variation = self.portfolio.copy()
            variation.sizing_strategy.update_target_positions(positions)
            variation.run_events()
            self.subsets["Target_Pos_" + str(positions)] = variation

    def trades_dict(self):
        trades = {}
        trades["Base"] = self.portfolio.trades
        for label, portfolio in self.subsets.items():
            trades[label] = portfolio.trades
        return trades

    def totals_dataframe(self):
        start = self.portfolio.trade_start_date
        df = DataFrame()
        df['Base'] = self.portfolio.summary[start:].Total
        for label, portfolio in self.subsets.items():
            raw_total = portfolio.summary[start:].Total
            starting_ratio = (df.Base.iloc[0] / raw_total.iloc[0])
            df[label] = raw_total * starting_ratio
        return df

    def plot_vs_base(self, **kwargs):
        df = self.totals_dataframe()
        df.plot(color = 'grey', **kwargs)
        df.Base.plot(color = 'blue')

    def plot_all(self, **kwargs):
        df = self.totals_dataframe()
        df.plot(**kwargs)


