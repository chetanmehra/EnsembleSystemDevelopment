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
from data_types.positions import Position, Returns
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

    def run(self):
        '''
        Run the strategy with the current settings.
        This will create the signals, trades, and base positions.
        Any filters specified will be applied also.
        '''
        self.generate_signals()
        self.apply_rules()
        self.apply_filters()
        
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
        self.trades = self.trades.apply_exit_condition(condition)

    def get_empty_dataframe(self, fill_data = None):
        '''
        Returns a dataframe with the same index as the market, and with
        columns equal to the tickers in the market.
        '''
        return self.market.get_empty_dataframe(fill_data)

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
        trade_timing = {"O" : "open", "C" : "close"}[self.trade_timing]
        return self.market.returns(trade_timing)

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

    def plot_returns(self, long_only = False, short_only = False, color = "blue", **kwargs):
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
# TODO add method to rerun a portfolio (e.g. after updating the position selection strategy).
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
        self.share_holdings = self.strategy.get_empty_dataframe(0) # Number of shares
        self.dollar_holdings = self.strategy.get_empty_dataframe(0) # Dollar value of positions
        self.positions = self.strategy.get_empty_dataframe(0) # Nominal positions
        self.summary = DataFrame(0, index = self.share_holdings.index, columns = ["Cash", "Holdings", "Total"])
        self.summary["Cash"] = starting_cash
        self.costs = DataFrame(0, index = self.share_holdings.index, columns = ["Commissions", "Slippage", "Total"])

        self.trade_size = (2000, 3500) # Min and max by dollar size
        self.rebalancing_strategy = NoRebalancing()
        vol_method = StdDevEMA(40)
        volatilities = vol_method(strategy.indicator_prices).shift(1)
        self.sizing_strategy = VolatilitySizingDecorator(0.2, volatilities, FixedNumberOfPositionsSizing(target_positions = 5))
        self.conditions = [MinimimumPositionSize()]
        self.position_checks = [PositionNaCheck(), PositionCostThreshold(0.02)]

        self.trades = None
        self.targets = strategy.positions.data.copy()
        self.running = False

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
        self.share_holdings = self.strategy.get_empty_dataframe(0) # Number of shares
        self.dollar_holdings = self.strategy.get_empty_dataframe(0) # Dollar value of positions
        self.positions = self.strategy.get_empty_dataframe(0) # Nominal positions
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
        If position_switching is True then the portfolio will aim to always be fully
        allocated by moving available funds into available open trades in the strategy
        entering new positions if required. 
        If this is False (default) then new positions are entered only when the 
        strategy triggers an entry event.
        '''
        self.running = True
        trading_days = self.share_holdings.index
        trade_prices = self.strategy.trade_prices
        #progress = ProgressBar(total = len(trading_days))
        for date in trading_days:
            strategy_events = self.strategy.events[date]
            # TODO doesn't the below mean that rebalancing won't work?
            if not len(strategy_events):
                continue
            positions = PositionChanges(self, date)
            # TODO below should be positions.apply(strategy_events)
            for event in strategy_events:
                event.update(positions)
            self.rebalancing_strategy(date, positions)
            positions.estimate_transactions()
            self.process_exits(positions)
            if position_switching:
                # Check available funds and request the top ranked positions
                # from the strategy.
                pass
            self.check_positions(positions)
            self.select(positions)
            transactions = Transactions(positions.num_shares, trade_prices.loc[date], date)
            self.apply(transactions)
            #progress.print(trading_days.get_loc(date))
        self.dollar_holdings = trade_prices.data * self.share_holdings
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        self.trades = Position(self.positions).create_trades(self.strategy)
        self.running = False
    
    def process_exits(self, positions):
        positions.select_exits()
        self.positions.loc[positions.date:, positions.exit_mask] = 0
  
    def check_positions(self, positions):
        for check in self.position_checks:
            check(self, positions)
    
    # TODO select needs to be a separate plug-in strategy
    def select(self, positions):
        # First apply all the trades which will reduce positions (i.e. increase cash)
        undecided_sells = ((positions.undecided_mask) &
                            (positions.suggested_size < positions.applied_size))
        positions.select(undecided_sells)
        # Select buys which will cost the least slippage + commissions as long
        # as there is cash available.
        perct_cost = positions.cost / positions.trade_size
        while True:
            undecided = positions.undecided_mask
            if not any(undecided):
                break
            lowest_cost = perct_cost[undecided].idxmin()
            if positions.trade_size[lowest_cost] >= (self.cash[positions.date] - positions.total_cost()):
                positions.deselect(lowest_cost)
            else:
                positions.select(lowest_cost)
                self.positions.loc[positions.date:, lowest_cost] = positions.suggested_size[lowest_cost] 
    
    def dollar_ratio(self, date):
        return self.sizing_strategy(self, date)

    def apply(self, transactions):
        date = transactions.date
        self.cash[date:] -= transactions.total_cost
        self.costs.loc[date:, "Commissions"] += transactions.total_commissions
        self.costs.loc[date:, "Slippage"] += transactions.total_slippage
        self.share_holdings[date:] += transactions.num_shares

    def run_trades(self):
        """
        Given a TradeCollection from a Strategy, determine the selected trades
        and resulting portfolio positions.
        """
        self.running = True
        trade_df = self.strategy.trades.as_dataframe().sort_values(by = 'entry')
        executed_trades = []
        for i in range(len(trade_df)): # Note index is unordered due to sort so need to create new index range for loop.
            trade = trade_df.iloc[i]
            trade_num = trade_df.index[i]
            if all(self.conditions_met_for(trade)):
                executed_trades.append(self.strategy.trades[trade_num])
                num_shares = Series(0, self.strategy.market.tickers)
                num_shares[trade.ticker] = int(min(self.cash[trade.entry], max(self.trade_size)) / trade.entry_price)
                trade_prices = self.strategy.trade_prices.loc[trade.entry]
                entry_txns = Transactions(num_shares, trade_prices, trade.entry)
                self.apply(entry_txns)
                trade_prices = self.strategy.trade_prices.loc[trade.exit]
                exit_txns = Transactions(-1 * num_shares, trade_prices, trade.exit)
                self.apply(exit_txns)
        self.trades = TradeCollection(executed_trades)
        self.dollar_holdings = self.share_holdings * self.strategy.trade_prices.data
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        self.running = False

    def conditions_met_for(self, trade):
        '''
        Returns a list of boolean values, one for each condition check specified for the portfolio.
        '''
        return [condition(self, trade) for condition in self.conditions]

    @property
    def starting_capital(self):
        return self.cash.iloc[0]

    @property
    def cash(self):
        '''
        Returns the series of dollar value of cash held over time.
        '''
        return self.summary["Cash"]

    @property
    def holdings_total(self):
        '''
        Returns the series of dollar value of all holdings (excluding cash).
        '''
        if self.running:
            prices = self.strategy.indicator_prices.shift(1)
            prices.iloc[0] = 0
            return (self.share_holdings * prices.data).sum(axis = 'columns')
        else:
            return self.summary["Holdings"]

    @property
    def value(self):
        '''
        Returns the series of total dollar value of the portfolio
        '''
        if self.running:
            return self.holdings_total + self.cash
        else:
            return self.summary["Total"]

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


class PositionChanges:
    
    def __init__(self, portfolio, date):
        self.current_shares = portfolio.share_holdings.loc[date] # assumes position updates are carried forward.
        # Get the prices from yesterday's indicator timing.
        self.prices = portfolio.strategy.indicator_prices.at(portfolio.strategy.trade_entry).loc[date]
        self.current_dollars = self.current_shares * self.prices
        self.dollar_ratio = portfolio.dollar_ratio(date)
        self.current_nominal = self.current_dollars / self.dollar_ratio
        # applied_size represents the nominal size that the portfolio has moved towards
        # this may be different from the target_size if, for example, an adjustment or entry wasn't 
        # acted upon.
        self.applied_size = portfolio.positions.loc[date] # assumes position updates are carried forward.
        self.target_size = portfolio.targets.shift(1).loc[date] # yesterday's strategy position size
        self.suggested_size = Series(index = self.current_shares.index)
        self.selected = Series(TradeSelected.NO, index = self.current_shares.index)
        self.type = Series("-", index = self.current_shares.index)
        self.date = date

    @property
    def exit_mask(self):
        return self.type == ExitEvent.Label

    @property
    def undecided_mask(self):
        return self.selected == TradeSelected.UNDECIDED

    @property
    def selected_mask(self):
        return self.selected == TradeSelected.YES

    def apply_rebalancing(self, tickers = None):
        if tickers is None:
            # All tickers which have not been touched by an event are rebalanced.
            tickers = np.isnan(self.suggested_size)
        self.suggested_size[tickers] = self.applied_size[tickers]
        self.type[tickers] = "rebalance"

    def estimate_transactions(self):
        num_shares = (self.dollar_ratio * self.suggested_size / self.prices)
        num_shares -= self.current_shares
        num_shares[np.isnan(num_shares)] = 0
        txns = Transactions(num_shares.astype(int), self.prices, self.date)
        self.cost = txns.slippage + txns.commissions
        self.trade_size = txns.dollar_positions
        self.num_shares = txns.num_shares

    def select(self, tickers):
        self.selected[tickers] = TradeSelected.YES

    def deselect(self, tickers):
        self.selected[tickers] = TradeSelected.NO
        self.trade_size[tickers] = 0
        self.num_shares[tickers] = 0
        self.cost[tickers] = 0
    
    def select_exits(self):
        self.select(self.exit_mask)

    def total_cost(self):
        return self.trade_size[self.selected_mask].sum() + self.cost[self.selected_mask].sum()
        

# REBALANCING STRATEGIES
class NoRebalancing:

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


# SIZING STRATEGIES
class FixedNumberOfPositionsSizing:

    def __init__(self, target_positions = 5):
        self.target_positions = target_positions

    def __call__(self, portfolio, date):
        nominal_dollar_position_size = portfolio.value[date] / self.target_positions
        return nominal_dollar_position_size

    def update_target_positions(self, target_positions):
        self.target_positions = target_positions


class VolatilitySizingDecorator:

    def __init__(self, vol_target, volatilities, base_sizing_strategy):
        self.target = vol_target
        self.volatilities = volatilities
        self.base_strategy = base_sizing_strategy

    def __call__(self, portfolio, date):
        volatility_ratio = self.target / self.volatilities.loc[date]
        base_size = self.base_strategy(portfolio, date)
        return volatility_ratio * base_size

    def update_target_positions(self, target_positions):
        self.base_strategy.update_target_positions(target_positions)


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
        self.num_shares[tickers] = 0
        self.dollar_positions[tickers] = 0
        self.slippage[tickers] = 0
        self.commissions[tickers] = 0

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
    
    
