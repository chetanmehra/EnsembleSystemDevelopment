# Signal analysis
# Analysis of entries, exits, and return characteristics over time.

import matplotlib.pyplot as plt
from pandas import qcut, DataFrame
from numpy import log


# Path dependencies
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