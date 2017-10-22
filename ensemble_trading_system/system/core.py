'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from pandas import DateOffset, Panel, DataFrame, Series
from enum import Enum

from utilities import ProgressBar
from system.interfaces import IndexerFactory
from data_types.trades import TradeCollection
from data_types.positions import Position, Returns
from measures.volatility import StdDevEMA


# TODO Strategy trade_timing should be just [O]pen or [C]lose, not OO/CC.
class Strategy:
    '''
    Strategy defines the base interface for Strategy objects
    '''
    def __init__(self, trade_timing, ind_timing):
        self.required_fields = ["market", "signal_generator", "position_rules"]
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
        self.trades = None

    def check_fields(self):
        missing_fields = []
        for field in self.required_fields:
            try:
                self.__getattribute__(field)
            except AttributeError:
                missing_fields += [field]
        if len(missing_fields):
            missing_fields = ", ".join(missing_fields)
            raise StrategyException("Missing parameters: {0}".format(missing_fields))
           
    def generate_signals(self):
        self.signal = self.signal_generator(self)

    def apply_rules(self):
        self.positions = self.position_rules(self)
        self.trades = self.positions.create_trades(self)

    def apply_filters(self):
        if len(self.filters) == 0:
            return
        for filter in self.filters:
            self.trades = filter(self)
        self.positions.update_from_trades(self.trades)

    def apply_filter(self, filter):
        '''
        Add a filter to a strategy which has already been run.
        This will run the filter, and add it to the list of 
        existing filters for the strategy.
        '''
        self.filters += [filter]
        self.trades = filter(self)
        self.positions.update_from_trades(self.trades)

    def apply_exit_condition(self, condition):
        '''
        Accepts an exit condition object e.g. StopLoss, which is
        passed to the Trade Collection to be applied to each trade.
        '''
        self.trades = self.trades.apply_exit_condition(condition)
        self.positions.update_from_trades(self.trades)
    
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
    
    def get_indicator_prices(self):
        '''
        Returns the prices dataframe relevant to the calculation of indicators.
        '''
        if self.ind_timing == "C":
            return self.market.close
        if self.ind_timing == "O":
            return self.market.open

    def get_trade_prices(self):
        '''
        Returns the dataframe of prices for trade timing.
        '''
        if self.trade_timing == "O":
            return self.market.open
        else:
            return self.market.close

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
        return createTrades(signal_data, self)

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

    # The below events methods are used in conjunction with Portfolio creation.  
    def define_events(self):
        pos_data = self.positions.data
        delta = pos_data - pos_data.shift(1)
        delta.iloc[0] = 0
        event_keys = self.get_empty_dataframe("-")
        # Note adjustment goes first and then gets overwritten by entry and exit
        event_keys[delta != 0] = "adjustment"
        event_keys[(delta != 0) & (delta == pos_data)] = "entry"
        event_keys[(delta != 0) & (pos_data == 0)] = "exit"
        self.events = {}
        self.events['key'] = event_keys
        self.events['sizes'] = delta
    
    def get_events(self, date):
        keys = self.events['key'].loc[date]
        event_tickers = self.events['key'].columns[keys != "-"]
        sizes = self.events['sizes'].loc[date]
        events = []
        for ticker in event_tickers:
            if keys[ticker] == "entry":
                events.append(EntryEvent(ticker, sizes[ticker]))
            elif keys[ticker] == "exit":
                events.append(ExitEvent(ticker, sizes[ticker]))
            elif keys[ticker] == "adjustment":
                events.append(AdjustmentEvent(ticker, sizes[ticker]))
        return events

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


# TODO Add Portfolio rebalancing methods.
# TODO Clean up unused Portfolio attributes and methods
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
        self.max_size = 2
        self.target_volatility = 0.2
        vol_method = StdDevEMA(40)
        self.volatilities = vol_method(strategy.get_indicator_prices()).shift(1)
        
        self.conditions = [MinimimumPositionSize()]
        self.transaction_checks = [TransactionCostThreshold(0.02)]
        self.position_checks = [PositionNaCheck()]
        
        self.current_positions = Series(0, index = self.positions.columns)
        self.targets = strategy.positions.data.copy()
        self.rebalancing_dates = []
        self.running = False


    def run_forecasts(self):
        """
        Runs the portfolio for the given strategy and starting conditions.
        This calculates the resulting portfolio based on positions sizes and forecast metrics.
        """
        self.running = True
        performance_metric = self.get_performance_metric()

        trading_days = self.share_holdings.index

        for date in trading_days:
            print(date)
            transactions = get_position_changes(date, performance_metric)
            transactions = self.check_transactions(transactions)
            self.apply(transactions)

        self.dollar_holdings = self.share_holdings * self.strategy.get_trade_prices()
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        #TODO Use a Positions object to calculate trades.
        self.running = False

    def get_performance_metric(self):
        """
        The performance metric is used to prioritise target positions.
        """
        current_fcst = self.strategy.signal.at(self.strategy.trade_entry)
        # If the forecasts are equal bias towards those with target position close to max_size.
        tie_breaker = (2 / (2 + ((self.strategy.positions.data - self.max_size) ** 2)))
        return current_fcst.data * tie_breaker


    def get_position_changes(self, date, performance_metric):
        """
        This method is responsible for selecting which positions to take from
        those the strategy has deemed are attractive, taking into account the
        target position sizes and diversification of the portfolio.
        The target positions are determined based on the performance metric, 
        and biased towards any existing positions through a cost factor.
        """
        cost_multiplier = 10
        pos_threshold = 0.1
        min_size = 0.5
        max_size = 2

        current_pos = self.positions.loc[date]
        target_pos = self.strategy.positions.loc[date]
        missing_targets = np.isnan(target_pos)
        target_pos[missing_targets] = current_pos[missing_targets]
        adj_target_pos = target_pos.copy()
        adj_target_pos[adj_target_pos > max_size] = max_size
        adj_target_pos[adj_target_pos < min_size] = 0
        fcst = performance_metric.loc[date]
        # TODO confirm if the forecast is known on the date in question, or if it should be lagged.
        full_res = fcst * target_pos - cost_multiplier * (target_pos - current_pos).abs()
        ix_full = full_res.sort_values(ascending = False).index

        full_selected = ix_full[adj_target_pos[ix_full].cumsum() <= self.target_size]

        selected_pos = Series(0, index = current_pos.index)
        selected_pos[full_selected] = adj_target_pos[full_selected]
        # TODO Positions are getting updated here before the transactions are checked.
        self.positions.loc[date:] = 0
        for ticker in adj_target_pos[full_selected][adj_target_pos[full_selected] > 0].index:
            self.positions.loc[date:, ticker] = adj_target_pos[ticker]

        ix_partial = ix_full[len(full_selected):]
        remaining = self.target_size - adj_target_pos[full_selected].sum()

        if len(ix_partial) and remaining >= min_size:
            partial_selected = ix_partial[0]
            selected_pos[partial_selected] = remaining
            self.positions.loc[date:, partial_selected] = remaining

        current_share_holdings = self.share_holdings.loc[date]
        current_prices = self.strategy.get_trade_prices().loc[date]
        current_dollar_holdings = current_share_holdings * current_prices
        nominal_dollar_size = (current_dollar_holdings.sum() + self.cash[date]) / self.target_size
        nominal_shares = nominal_dollar_size / current_prices
        target_shares = selected_pos * nominal_shares
        share_movements = target_shares - current_share_holdings
        share_movements[np.isnan(share_movements)] = 0
        
        return Transactions(share_movements.astype(int, copy = False), current_prices, date)

    def run_events(self):
        '''
        This approach gets the series of events from the strategy, and the portfolio can 
        also add its own events (e.g. rebalancing). Portfolio events are always applied 
        after the Strategy events (i.e. can be used to overrule Strategy events).
        The events are turned into Transactions for execution.
        '''
        self.running = True
        self.strategy.define_events()
        trading_days = self.share_holdings.index
        progress = ProgressBar(total = len(trading_days))
        for date in trading_days:
            strategy_events = self.strategy.get_events(date)
            if not len(strategy_events):
                continue
            positions = self.get_positions(date)
            for event in strategy_events:
                event.update(positions)
            if self.rebalancing(date):
                self.apply_rebalancing(positions)
            self.estimate_transactions(positions)
            self.process_exits(positions)
            self.check_positions(positions)
            self.select(positions)
            trade_prices = self.strategy.get_trade_prices().loc[date]
            transactions = Transactions(positions.num_shares, trade_prices, date)
            self.apply(transactions)
            progress.print(trading_days.get_loc(date))
        self.dollar_holdings = self.share_holdings * self.strategy.get_trade_prices()
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        self.running = False

    def get_positions(self, date):
        positions = DataFrame(index = self.share_holdings.columns)
        positions['current_shares'] = self.share_holdings.loc[date] # assumes position updates are carried forward.
        # Get the prices from yesterday's indicator timing.
        # TODO currently assumes indicator at Close and trade at Open, remove manual shift and use data object timing method.
        positions['prices'] = self.strategy.get_indicator_prices().shift(1).loc[date]
        positions['current_dollars'] = positions.current_shares * positions.prices
        positions['dollar_ratio'] = self.dollar_ratio(date)
        positions['current_nominal'] = positions.current_dollars / positions.dollar_ratio
        # applied_size represents the nominal size that the portfolio has moved towards
        # this may be different from the target_size if, for example, an adjustment or entry wasn't 
        # acted upon.
        positions['applied_size'] = self.positions.loc[date] # assumes position updates are carried forward.
        positions['target_size'] = self.targets.shift(1).loc[date] # yesterday's strategy position size
        positions['suggested_size'] = np.nan
        positions['selected'] = TradeSelected.NO
        positions['type'] = "-"
        positions.date = date
        return positions
    
    # Below would require a list of rebalancing dates to be set up on Portfolio initialisation
    # This could even be set up with a plugin rebalancing strategy - e.g. not just at fixed dates,
    # but if the positions get out of line by a certain percentage.
    def rebalancing(self, date):
        return date in self.rebalancing_dates
  
    def set_weekly_rebalancing(self, day = "FRI"):
        # Refer to the following link for options:
        # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        # Note the count method is only used to stop pandas from throwing a warning, as
        # it wants to know how to aggregate the data.
        self.rebalancing_dates = self.value.resample('W-' + day).count().index

    def set_month_end_rebalancing(self):
        self.rebalancing_dates = self.value.resample('M').count().index

    def apply_rebalancing(self, positions):
        unmodified = np.isnan(positions.suggested_size)
        positions.suggested_size[unmodified] = positions.applied_size[unmodified]
        positions.loc[unmodified, 'type'] = "rebalance"

    def estimate_transactions(self, positions):
        num_shares = (positions.dollar_ratio * positions.suggested_size / positions.prices)
        num_shares -= positions.current_shares
        num_shares[np.isnan(num_shares)] = 0
        txns = Transactions(num_shares.astype(int), positions.prices, positions.date)
        positions["cost"] = txns.slippage + txns.commissions
        # TODO trade_size does not include slippage and commissions - messes up selection procedure
        positions["trade_size"] = txns.dollar_positions
        positions["num_shares"] = txns.num_shares
    
    def process_exits(self, positions):
        positions.loc[positions.type == ExitEvent.Label, 'selected'] = TradeSelected.YES
        self.positions.loc[positions.date:, positions.type == ExitEvent.Label] = 0
  
    def check_positions(self, positions):
        # This is already a check_transactions method set up in Portfolio
        # Need to modify checks to work with positions.
        for check in self.position_checks:
            check(self, positions)
      
    def select(self, positions):
        # First apply all the trades which will reduce positions (i.e. increase cash)
        undecided_sells = ((positions.selected == TradeSelected.UNDECIDED) &
                            (positions.suggested_size < positions.applied_size))
        positions.loc[undecided_sells, 'selected'] = TradeSelected.YES
        # Select buys which will cost the least slippage + commissions as long
        # as there is cash available.
        perct_cost = positions.cost / positions.trade_size
        while True:
            undecided = (positions.selected == TradeSelected.UNDECIDED)
            if not any(undecided):
                break
            selected = (positions.selected == TradeSelected.YES)
            lowest_cost = perct_cost[undecided].idxmin()
            if positions.trade_size[lowest_cost] >= (self.cash[positions.date] - positions.trade_size[selected].sum()):
                positions.loc[lowest_cost, 'selected'] = TradeSelected.NO
                positions.loc[lowest_cost, 'trade_size'] = 0
                positions.loc[lowest_cost, 'num_shares'] = 0
                positions.loc[lowest_cost, 'cost'] = 0

            else:
                positions.loc[lowest_cost, 'selected'] = TradeSelected.YES
                self.positions.loc[positions.date:, lowest_cost] = positions.suggested_size[lowest_cost] 
        
    
    def dollar_ratio(self, date):
        nominal_dollar_position_size = self.value[date] / self.target_size
        volatility_ratio = self.target_volatility / self.volatilities.loc[date]
        return volatility_ratio * nominal_dollar_position_size

    @property
    def target_size(self):
        """
        Returns the target size (scalar) for the portfolio in number of positions.
        Changing this method will allow different behaviour such as setting the 
        portfolio to hold a set number of positions through time, or to change 
        positions as the the portfolio value increases / decreases.
        """
        return 5    
        
    def check_transactions(self, target_transactions):
        """
        This runs through a series of checks, removing any transactions that don't
        meet the conditions (e.g. if cost would be too high).
        """
        for check in self.transaction_checks:
            target_transactions = check(self, target_transactions)
        return target_transactions

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
                trade_prices = self.strategy.get_trade_prices().loc[trade.entry]
                entry_txns = Transactions(num_shares, trade_prices, trade.entry)
                self.apply(entry_txns)
                trade_prices = self.strategy.get_trade_prices().loc[trade.exit]
                exit_txns = Transactions(-1 * num_shares, trade_prices, trade.exit)
                self.apply(exit_txns)
        self.trades = TradeCollection(executed_trades)
        self.dollar_holdings = self.share_holdings * self.strategy.get_trade_prices()
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
            prices = self.strategy.get_indicator_prices().shift(1)
            prices.iloc[0] = 0
            return (self.share_holdings * prices).sum(axis = 'columns')
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


# Portfolio Transaction Acceptance Criteria
class TransactionCostThreshold:

    def __init__(self, cost_threshold = 0.02):
        self.cost_threshold = cost_threshold

    def __call__(self, portfolio, transactions):
        cost_ratio = ((transactions.commissions + transactions.slippage) / transactions.dollar_positions).abs()
        current_shares = portfolio.share_holdings.loc[transactions.date]
        net_size = current_shares + transactions.num_shares
        costly = (cost_ratio > self.cost_threshold)
        not_exits = (net_size != 0)
        costly_tickers = transactions.tickers[costly & not_exits]
        transactions.remove(costly_tickers)
        return transactions


class TradeEvent:

    def __init__(self, ticker, size):
        self.ticker = ticker
        self.size = size

    def update(self, positions):
        positions.type.loc[self.ticker] = self.Label
        positions.loc[self.ticker, 'selected'] = TradeSelected.UNDECIDED

class EntryEvent(TradeEvent):
    Label = "entry"

    def update(self, positions):
        positions.target_size.loc[self.ticker] = self.size
        positions.suggested_size.loc[self.ticker] = self.size
        super().update(positions)
        

class ExitEvent(TradeEvent):
    Label = "exit"

    def update(self, positions):
        positions.target_size.loc[self.ticker] = 0
        positions.suggested_size.loc[self.ticker] = 0
        super().update(positions)
        

class AdjustmentEvent(TradeEvent):
    Label = "adjustment"

    def update(self, positions):
        positions.target_size.loc[self.ticker] += self.size
        if positions.applied_size.loc[self.ticker] != 0:
            positions.suggested_size.loc[self.ticker] = positions.target_size[self.ticker]
            super().update(positions)
        

  
class TradeSelected(Enum):

    UNDECIDED = 0
    YES = 1
    NO = -1


# These checks can be used to control the position selection process
# If the check fails the position change is removed from consideration.
class PositionNaCheck:

    def __call__(self, portfolio, positions):
        nans = np.isnan(positions.trade_size)
        positions.loc[nans, 'selected'] = TradeSelected.NO
        zeroes = positions.trade_size == 0
        positions.loc[zeroes, 'selected'] = TradeSelected.NO


class PositionMaxSize:
  
    def __init__(self, size):
        self.limit = size

    def __call__(self, portfolio, positions):
        exceeding_limit = positions.trade_size > self.limit
        positions.loc[exceeding_limit, 'selected'] = TradeSelected.NO
      
class PositionMinSize:

    def __init__(self, size):
        self.limit = size
        
    def __call__(self, portfolio, positions):
        exceeding_limit = positions.trade_size < self.limit
        positions.loc[exceeding_limit, 'selected'] = TradeSelected.NO

class PositionCostThreshold:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, portfolio, positions):
        perct_cost = positions.cost / positions.trade_size
        exceeding_limit = perct_cost > self.threshold
        positions.loc[exceeding_limit, 'selected'] = TradeSelected.NO
    
    
