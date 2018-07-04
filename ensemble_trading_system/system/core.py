'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from pandas import DateOffset, Panel, DataFrame, Series

from system.interfaces import IndexerFactory
from data_types.trades import TradeCollection
from data_types.positions import Position
from data_types.returns import Returns
from data_types.constants import TradeSelected
from data_types.events import EventCollection, ExitEvent
from measures.volatility import StdDevEMA


class Strategy:
    '''
    Strategy manages creation of positions and trades for a given
    strategy idea and plotting of results.
    '''
    def __init__(self, trade_timing, ind_timing):
        self.name = None
        self.market = None
        # Calculation results
        self.signal = None
        self.positions = None
        # Component methods
        self.signal_generator = None
        self.position_rules = None
        self.filters = []
        # Timing parameters
        self.indexer = IndexerFactory(trade_timing, ind_timing)
        self.decision = self.indexer("decision")
        self.trade_entry = self.indexer("trade")
        self.open = self.indexer("open")
        self.close = self.indexer("close")

    def __str__(self):
        if self.name is not None:
            first_line = '{}\n'.format(self.name)
        else:
            first_line = ''
        second_line = 'Signal:\t{}\n'.format(self.signal_generator.name)
        third_line = 'Position Rule:\t{}\n'.format(self.position_rules.name)
        if len(self.filters) > 0:
            fourth_line = 'Filter:\t{}\n'.format('\n'.join([f.name for f in self.filters]))
        else:
            fourth_line = 'No filter specified.\n'
        return first_line + second_line + third_line + fourth_line

    @property
    def trade_timing(self):
        '''
        Return the timing (open, close) for when trades occur.
        '''
        return self.indexer.trade_timing
    
    @property
    def ind_timing(self):
        '''
        Return the timing (open, close) for when calculations to determine
        actions are performed.
        '''
        return self.indexer.ind_timing

    @property
    def trades(self):
        return self.positions.trades

    @trades.setter
    def trades(self, trades):
        self.positions.update_from_trades(trades)

    @property
    def events(self):
        return self.positions.events

    @property
    def prices(self):
        return self.market
    
    @property
    def indicator_prices(self):
        return self.prices.at(self.decision)

    @property
    def trade_prices(self):
        return self.prices.at(self.trade_entry)

    def copy(self):
        strat_copy = Strategy(self.trade_timing, self.ind_timing)
        strat_copy.market = self.market
        strat_copy.signal_generator = self.signal_generator
        strat_copy.position_rules = self.position_rules
        strat_copy.filters = self.filters
        return strat_copy

    def run(self):
        '''
        Run the strategy with the current settings.
        This will create the signals, trades, and base positions.
        Any filters specified will be applied also.
        '''
        self.generate_signals()
        self.apply_rules()
        self.apply_filters()
        self.rebase()
        
    def rerun(self):
        '''
        Run the strategy by first clearing any previous calculated data.
        Typically used when the strategy has already been run, but changes have been
        made to the settings
        '''
        self.reset()
        self.run()

    def reset(self):
        '''
        Clears all current results from the Strategy.
        '''
        self.signal = None
        self.positions = None

    def rebase(self):
        '''
        rebase makes a copy of the positions (and trades) as base reference.
        Any further changes to these can then be compared with the base result.
        '''
        self.base_positions = self.positions
        
    def generate_signals(self):
        self.signal = self.signal_generator(self)

    def apply_rules(self):
        self.positions = self.position_rules(self)
        self.positions.trades = self.positions.create_trades(self)

    def apply_filters(self):
        if len(self.filters) == 0:
            return
        for filter in self.filters:
            self.trades = filter(self)

    def apply_filter(self, filter):
        '''
        Add a filter to a strategy which has already been run.
        This will run the filter, and add it to the list of 
        existing filters for the strategy.
        '''
        self.filters += [filter]
        self.trades = filter(self)

    def apply_exit_condition(self, condition):
        '''
        Accepts an exit condition object e.g. StopLoss, which is
        passed to the Trade Collection to be applied to each trade.
        '''
        self.trades = self.trades.apply(condition)

    def get_empty_dataframe(self, fill_data = None):
        '''
        Returns a dataframe with the same index as the market, and with
        columns equal to the tickers in the market.
        '''
        return self.market.get_empty_dataframe(fill_data)

    def subset(self, subset_tickers):
        '''
        subset supports cross validation across market constituents.
        The idea being multiple subsets are taken and performance is 
        compared to determine sensitivity of the strategy to the tickers
        included in the market.
        '''
        new_strat = Strategy(self.trade_timing, self.ind_timing)
        new_strat.market = self.market.subset(subset_tickers)
        new_strat.positions = self.positions.subset(subset_tickers)
        return new_strat

    def buy_and_hold_trades(self):
        '''
        Return the TradeCollection assuming all tickers are bought and held
        for the duration of the test period.
        '''
        signal_data = self.get_empty_dataframe()
        signal_data[:] = 1
        return Position(signal_data).create_trades(self)

    @property
    def market_returns(self):
        '''
        Gets the dataframe of market returns relevant for the trade timing.
        '''
        return self.market.at(self.trade_entry).returns()

    @property
    def returns(self):
        '''
        Gets the dataframe of strategy returns, i.e. positions applied to the market
        returns.
        '''
        return self.positions.applied_to(self.market_returns)
    
    @property
    def long_returns(self):
        '''
        Gets the strategy returns for long trades only.
        '''
        positions = self.positions.long_only()
        return positions.applied_to(self.market_returns)
    
    @property
    def short_returns(self):
        '''
        Gets the strategy returns for short trades only.
        '''
        positions = self.positions.short_only()
        return positions.applied_to(self.market_returns)

    # Reporting methods
    def plot_measures(self, ticker, start, end, ax):
        self.signal.plot_measures(ticker, start, end, ax)

    def plot_result(self, long_only = False, short_only = False, color = "blue", **kwargs):
        '''
        Plots the returns of the strategy vs the market returns.
        '''
        if long_only:
            returns = self.long_returns
        elif short_only:
            returns = self.short_returns
        else:
            returns = self.returns
        start = self.positions.start
        returns.plot(start = start, color = color, label = "Strategy", **kwargs)
        self.market_returns.plot(start = start, color = "black", label = "Market")
        plt.legend(loc = "upper left")

    def plot_trade(self, key):
        '''
        Given the key of a trade (trade number, or ticker), this will produce a candlestick plt, 
        along with the measures for the trade, entry, and exit.
        '''
        trades = self.trades[key]
        if isinstance(trades, list):
            if len(trades) == 0:
                print('No trades for {}'.format(key))
                return
            entries, exits = zip(*[(T.entry, T.exit) for T in trades])
            entries = list(entries)
            exits = list(exits)
            ticker = key
        else:
            entries = [trades.entry]
            exits = [trades.exit]
            ticker = trades.ticker
        start = min(entries) - DateOffset(10)
        end = max(exits) + DateOffset(10)
        fig, ax = self.market.candlestick(ticker, start, end)
        self.plot_measures(ticker, start, end, ax)
        for filter in self.filters:
            filter.plot(ticker, start, end, ax)
        lo_entry = self.market.low[ticker][entries] * 0.95
        hi_exit = self.market.high[ticker][exits] * 1.05
        plt.scatter(entries, lo_entry, marker = '^', color = 'green')
        plt.scatter(exits, hi_exit, marker = 'v', color = 'red')
        plt.title(ticker)


# TODO Clean up unused Portfolio attributes and methods
# TODO consider merging Transactions and PositionChanges
# TODO find a better way to update Portfolio positions from PositionChanges
class Portfolio:
    '''
    The portfolio class manages cash and stock positions, as well as rules 
    regarding the size of positions to hold and rebalancing.
    The portfolio contains a list of trade conditions. Each one is a function which 
    accepts the portfolio, and trade, and returns a boolean. Each one of the conditions
    must return True for the trade to be accepted.
    '''
    def __init__(self, strategy, starting_cash):
        self.strategy = strategy
        self.share_holdings = self.strategy.get_empty_dataframe() # Number of shares
        self.dollar_holdings = self.strategy.get_empty_dataframe(0) # Dollar value of positions
        self.positions = self.strategy.get_empty_dataframe() # Nominal positions
        self.position_states = PositionState(self.share_holdings.columns, starting_cash)
        self.summary = DataFrame(0, index = self.share_holdings.index, columns = ["Cash", "Holdings", "Total"])
        self.summary["Cash"] = starting_cash
        self.costs = DataFrame(0, index = self.share_holdings.index, columns = ["Commissions", "Slippage", "Total"])

        self.rebalancing_strategy = NoRebalancing()
        self.sizing_strategy = SizingStrategy()
        volatilities = StdDevEMA(40)(strategy.indicator_prices.at(strategy.trade_entry))
        self.sizing_strategy.multipliers.append(VolatilityMultiplier(0.2, volatilities))
        self.position_checks = [PositionNaCheck(), PositionCostThreshold(0.02)]
        
        self.trades = None
        self.targets = strategy.positions.data.copy()

    def copy(self):
        port_copy = Portfolio(self.strategy, self.starting_capital)
        port_copy.position_checks = self.position_checks
        port_copy.sizing_strategy = self.sizing_strategy
        return port_copy

    def reset(self):
        '''
        reset - Sets the Portfolio back to its starting condition ready to be run again.
        '''
        starting_cash = self.starting_capital
        self.share_holdings = self.strategy.get_empty_dataframe() # Number of shares
        self.dollar_holdings = self.strategy.get_empty_dataframe(0) # Dollar value of positions
        self.positions = self.strategy.get_empty_dataframe() # Nominal positions
        self.position_states = PositionState(self.share_holdings.columns, starting_cash)
        self.summary = DataFrame(0, index = self.share_holdings.index, columns = ["Cash", "Holdings", "Total"])
        self.summary["Cash"] = starting_cash
        self.costs = DataFrame(0, index = self.share_holdings.index, columns = ["Commissions", "Slippage", "Total"])
        self.trades = None

    def change_starting_capital(self, starting_cash):
        self.reset()
        self.summary["Cash"] = starting_cash

    def run(self, position_switching = False):
        '''
        This approach gets the series of events from the strategy, and the portfolio can 
        also add its own events (e.g. rebalancing). Portfolio events are always applied 
        after the Strategy events (i.e. can be used to overrule Strategy events).
        The events are turned into Transactions for execution.
        '''
        trading_days = self.share_holdings.index
        trade_prices = self.strategy.trade_prices
        #progress = ProgressBar(total = len(trading_days))

        for date in trading_days:
            strategy_events = self.strategy.events[date]
            # TODO rebalancing strategy needs to add events to the queue.
            #if date in self.rebalancing_strategy.rebalance_dates:
            #    self.rebalancing_strategy(strategy_events)
            if not len(strategy_events):
                continue
            self.position_states.update(self, date)
            self.position_states.apply(strategy_events)
            self.position_states.estimate_transactions()
            self.process_exits(self.position_states)
            self.check_positions(self.position_states)
            self.select(self.position_states)
            self.apply(self.position_states)
            #progress.print(trading_days.get_loc(date))

        self.positions = Position(self.positions.ffill().fillna(0))
        self.trades = self.positions.create_trades(self.strategy)
        self.share_holdings = self.share_holdings.ffill().fillna(0)
        self.dollar_holdings = trade_prices.data * self.share_holdings
        self.dollar_holdings = self.dollar_holdings.ffill()
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        

    def process_exits(self, positions):
        positions.select_exits()
        self.positions.loc[positions.date, positions.exit_mask] = 0
  
    def check_positions(self, positions):
        for check in self.position_checks:
            check(self, positions)
    
    # TODO select needs to be a separate plug-in strategy
    def select(self, positions):
        # First apply all the trades which will reduce positions (i.e. increase cash)
        undecided_sells = ((positions.undecided_mask) &
                            (positions.target_size < positions.applied_size))
        positions.select(undecided_sells)
        # Select buys which will cost the least slippage + commissions as long
        # as there is cash available.
        perct_cost = positions.cost / positions.trade_size
        while True:
            undecided = positions.undecided_mask
            if not any(undecided):
                break
            lowest_cost = perct_cost[undecided].idxmin()
            if positions.trade_size[lowest_cost] >= (positions.available_cash()):
                positions.deselect(lowest_cost)
            else:
                positions.select(lowest_cost)
                self.positions.loc[positions.date, lowest_cost] = positions.target_size[lowest_cost] 
        positions.finalise()
    
    def dollar_ratio(self, date):
        return self.sizing_strategy(self, date)

    def apply(self, positions):
        date = positions.date
        self.cash.loc[date:] -= positions.txns.total_cost
        self.costs.loc[date:, "Commissions"] += positions.txns.total_commissions
        self.costs.loc[date:, "Slippage"] += positions.txns.total_slippage
        self.share_holdings.loc[date] = positions.current_shares
        
    @property
    def starting_capital(self):
        return self.cash.iloc[0]

    @property
    def cash(self):
        '''
        Returns the series of dollar value of cash held over time.
        '''
        return self.summary.loc[:, "Cash"]

    @property
    def holdings_total(self):
        '''
        Returns the series of dollar value of all holdings (excluding cash).
        '''
        return self.summary["Holdings"]

    @property
    def value(self):
        '''
        Returns the series of total dollar value of the portfolio
        '''
        return self.summary["Total"]

    @property
    def current_value(self):
        '''
        Returns the current value of the portfolio
        Typically used while the portfolio is performing calculations.
        '''
        return self.position_states.current_dollars.sum() + self.position_states.current_cash

    @property
    def value_ex_cost(self):
        '''
        Returns the series of total dollar value assuming zero costs
        '''
        return self.value + self.costs["Total"]

    @property
    def returns(self):
        '''
        Returns the daily returns for the portfolio
        '''
        returns = self.value / self.value.shift(1) - 1
        returns[0] = 0
        return Returns(returns)

    @property
    def returns_ex_cost(self):
        '''
        Returns the daily returns for the portfolio assuming zero costs
        '''
        returns = self.value_ex_cost / self.value_ex_cost.shift(1) - 1
        returns[0] = 0
        return Returns(returns)

    @property
    def cumulative_returns(self):
        '''
        Returns the cumulative returns series for the portfolio
        '''
        return self.returns.cumulative()

    @property
    def drawdowns(self):
        '''
        Returns the calculated drawdowns and highwater series for the portfolio returns.
        '''
        return self.returns.drawdowns()

    def max_position_percent(self):
        pct_holdings = self.dollar_holdings.divide(self.value, axis = 0)
        return pct_holdings.max(axis = 1)

    @property
    def trade_start_date(self):
        return self.dollar_holdings.index[self.dollar_holdings.sum(axis = 1) > 0][0]

    def plot_result(self, dd_ylim = None, rets_ylim = None):
        '''
        Plots the portfolio returns and drawdowns vs the market.
        '''
        start = self.trade_start_date
        f, axarr = plt.subplots(2, 1, sharex = True)
        axarr[0].set_ylabel('Return')
        axarr[1].set_ylabel('Drawdown')
        axarr[1].set_xlabel('Days in trade')
        self.cumulative_returns[start:].plot(ax = axarr[0], ylim = rets_ylim)
        self.returns_ex_cost.cumulative()[start:].plot(ax = axarr[0], color = 'cyan', linewidth = 0.5)
        dd = self.drawdowns
        dd.Highwater[start:].plot(ax = axarr[0], color = 'red')
        dd.Drawdown[start:].plot(ax = axarr[1], ylim = dd_ylim)
        mkt_returns = self.strategy.market_returns
        mkt_dd = mkt_returns.drawdowns().Drawdown
        mkt_returns.plot(start = start, ax = axarr[0], color = 'black')
        mkt_dd[start:].plot(ax = axarr[1], color = 'black')
        return axarr


class PositionState:
    
    def __init__(self, tickers, current_cash):
        self.current_cash = current_cash
        self.current_shares = Series(0, index = tickers)
        self.applied_size = Series(0, index = tickers)
        self.target_size = Series(index = tickers, dtype = float)
        self.selected = Series(TradeSelected.NO, index = tickers)
        self.type = Series("-", index = tickers)
        self.pending_events = [] # events which were not actioned (e.g. due to missing price data)

    def update(self, portfolio, date):
        # Get the prices from yesterday's indicator timing.
        self.prices = portfolio.strategy.indicator_prices.at(portfolio.strategy.trade_entry).loc[date]
        self.current_dollars = self.current_shares * self.prices
        self.dollar_ratio = portfolio.dollar_ratio(date)
        self.current_nominal = self.current_dollars / self.dollar_ratio
        self.target_size = Series(index = self.current_shares.index)
        self.selected = Series(TradeSelected.NO, index = self.current_shares.index)
        self.type = Series("-", index = self.current_shares.index)
        self.date = date

    def finalise(self):
        self.current_shares += self.txns.num_shares
        self.current_cash -= self.txns.total_cost
        # applied_size represents the nominal size that the portfolio has moved towards
        # this may be different from the target_size if, for example, an adjustment or entry wasn't 
        # acted upon.
        self.applied_size.loc[self.selected_mask] = self.target_size[self.selected_mask]

    @property
    def exit_mask(self):
        return self.type == ExitEvent.Label

    @property
    def undecided_mask(self):
        return self.selected == TradeSelected.UNDECIDED

    @property
    def selected_mask(self):
        return self.selected == TradeSelected.YES

    def apply(self, events):
        self.pending_events.extend(events)
        all_events = self.pending_events
        self.pending_events = []
        for event in all_events:
            if np.isnan(self.prices[event.ticker]):
                self.pending_events.append(event)
            else:
                event.update(self)

    def apply_rebalancing(self, tickers = None):
        if tickers is None:
            # All tickers which have not been touched by an event are rebalanced.
            tickers = np.isnan(self.target_size)
        self.target_size.loc[tickers] = self.applied_size[tickers]
        #TODO below should use a RebalanceEvent.Label not string.
        self.type.loc[tickers] = "rebalance"

    def estimate_transactions(self):
        num_shares = (self.dollar_ratio * self.target_size / self.prices)
        num_shares -= self.current_shares
        num_shares[np.isnan(num_shares)] = 0
        self.txns = Transactions(num_shares.astype(int), self.prices, self.date)

    @property
    def cost(self):
        return self.txns.slippage + self.txns.commissions

    @property
    def trade_size(self):
        return self.txns.dollar_positions

    @property
    def num_shares(self):
        return self.txns.num_shares

    def select(self, tickers):
        self.selected.loc[tickers] = TradeSelected.YES

    def deselect(self, tickers):
        self.selected.loc[tickers] = TradeSelected.NO
        self.trade_size.loc[tickers] = 0
        self.num_shares.loc[tickers] = 0
        self.cost.loc[tickers] = 0
    
    def select_exits(self):
        self.select(self.exit_mask)

    def available_cash(self):
        return self.current_cash - self.total_cost()

    def total_cost(self):
        return self.trade_size[self.selected_mask].sum() + self.cost[self.selected_mask].sum()


class SizingStrategy:
    '''
    The SizingStrategy is responsible for providing the dollar ratio to turn
    the strategy's nominal size into the size in dollars.
    There are basically two components:
        1. The base dollar size - this is basically determined by the target
        number of positions to be held (i.e. capital / number of positions)
        2. Multipliers - there can be an arbitrary number of multipliers which
        modify the size for each instrument (e.g. for volatility scaling)
    The strategy aims to increase the number of positions as capital allows. 
    The rate at which positions are added is dictated by the
    diversifier parameter which must be in the range 0-1. A diversifier close 
    to zero will hold the minimum number of positions possible while still
    satisfying the maximum size constraint (as a proportion of total capital).
    This would therefore act as targeting a fixed number of positions (1 / max_size). 
    A diversifier close to 1 will increase the number of positions as soon as 
    the minimum position size constraint can be met (as a dollar size).
    '''
    def __init__(self, diversifier = 0.5, min_size = 2500, max_size = 0.3):
        self.diversifier = diversifier
        self.min_size = min_size
        self.max_size = max_size
        self.multipliers = []

    def __call__(self, portfolio, date):
        dollar_ratio = self.dollar_ratio(portfolio, date)
        for multiplier in self.multipliers:
            dollar_ratio *= multiplier(portfolio, date)
        return dollar_ratio

    def dollar_ratio(self, portfolio, date):
        max_num_positions = portfolio.current_value / self.min_size
        min_num_positions = (1 / self.max_size)
        target = min_num_positions + self.diversifier * (max_num_positions - min_num_positions)
        # Note: we convert target to an int to round down the target number
        # of positions, and bump up the nominal dollar size to provide some 
        # leeway for multipliers which may want to reduce the size.
        return portfolio.current_value / int(target)


class VolatilityMultiplier:
    '''
    Adjusts the position multiplier based on the current volatility
    up or down towards the specified target.
    '''

    def __init__(self, vol_target, volatilities):
        self.target = vol_target
        self.volatilities = volatilities
        self.ratio = vol_target / volatilities

    def __call__(self, portfolio, date):
        return self.ratio.loc[date]


# REBALANCING STRATEGIES
class NoRebalancing:

    def __init__(self):
        self.rebalancing_dates = []

    def __call__(self, positions, date):
        pass


class FixedRebalanceDates:

    def __init__(self):
        self.rebalancing_dates = []
  
    def set_weekly_rebalancing(self, daily_series, day = "FRI"):
        # Refer to the following link for options:
        # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        # Note the count method is only used to stop pandas from throwing a warning, as
        # it wants to know how to aggregate the data.
        self.rebalancing_dates = daily_series.resample('W-' + day).count().index

    def set_month_end_rebalancing(self, daily_series):
        self.rebalancing_dates = daily_series.resample('M').count().index

    def __call__(self, positions, date):
        if date in self.rebalancing_dates:
            positions.apply_rebalancing()


# Portfolio Trade Acceptance Criteria
class MinimimumPositionSize:

    def __call__(self, portfolio, trade):
        '''
        Provides a trade condition where the cash at the time of entry must be greater
        than the minimum allowable portfolio size.
        '''
        return portfolio.cash[trade.entry] > min(portfolio.trade_size)

class MinimumTradePrice:

    def __init__(self, min_price):
        self.min_price = min_price

    def __call__(self, portfolio, trade):
        """
        Accepts trades provided the entry price is above a defined threshold.
        The idea is to stop positions taken in shares which would incur excessive slippage.
        """
        return trade.entry_price >= min_price


class Transactions:
    """
    Transactions holds a series of buy and sell transactions for each ticker for a particular day.
    The Transactions object can then calculate the total cash flows, and position movements for 
    the Portfolio.
    """
    def __init__(self, num_shares, prices, date):
        self.date = date
        nominal_size = num_shares * prices
        trim_factor = 0.02 # This reduces the size of purchases to account for transaction costs
        trim_size = (nominal_size * trim_factor) / prices
        trim_size[np.isnan(trim_size)] = 0
        trim_size = trim_size.astype(int) + 1 # Round to ceiling
        sales = num_shares <= 0
        trim_size[sales] = 0
        num_shares -= trim_size
        self.num_shares = num_shares
        self.tickers = num_shares.index
        self.dollar_positions = num_shares * prices
        self.slippage = self.estimate_slippage(prices, num_shares)
        self.commissions = Series(0, self.tickers)
        self.commissions[num_shares != 0] = 11 # $11 each way
    
    def estimate_slippage(self, prices, num_shares):
        """
        Estimates the slippage assuming that the spread is at least one price increment wide.
        Price increments used are as defined by the ASX:
            Up to $0.10     0.1c
            Up to $2.00     0.5c
            $2.00 and over  1c
        """
        slippage = Series(0, self.tickers)
        slippage[prices < 0.1] = num_shares[prices < 0.1] * 0.001
        slippage[(prices >= 0.1) & (prices < 2.0)] = num_shares[(prices >= 0.1) & (prices < 2.0)] * 0.005
        slippage[prices >= 2.0] = num_shares[prices >= 2.0] * 0.005
        return abs(slippage)

    def remove(self, tickers):
        self.num_shares.loc[tickers] = 0
        self.dollar_positions.loc[tickers] = 0
        self.slippage.loc[tickers] = 0
        self.commissions.loc[tickers] = 0

    @property
    def active(self):
        return self.tickers[self.num_shares != 0]

    @property
    def total_cost(self):
        return self.dollar_positions.sum() + self.total_commissions + self.total_slippage

    @property
    def total_commissions(self):
        return self.commissions.sum()

    @property
    def total_slippage(self):
        return self.slippage.sum()



# These checks can be used to control the position selection process
# If the check fails the position change is removed from consideration.
class PositionNaCheck:

    def __call__(self, portfolio, positions):
        nans = np.isnan(positions.trade_size)
        positions.deselect(nans)
        zeroes = positions.trade_size == 0
        positions.deselect(zeroes)


class PositionMaxSize:
  
    def __init__(self, size):
        self.limit = size

    def __call__(self, portfolio, positions):
        exceeding_limit = positions.trade_size > self.limit
        positions.deselect(exceeding_limit)
      
class PositionMinSize:

    def __init__(self, size):
        self.limit = size
        
    def __call__(self, portfolio, positions):
        exceeding_limit = positions.trade_size < self.limit
        positions.deselect(exceeding_limit)

class PositionCostThreshold:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, portfolio, positions):
        perct_cost = positions.cost / positions.trade_size
        exceeding_limit = perct_cost > self.threshold
        positions.deselect(exceeding_limit)
    
    
