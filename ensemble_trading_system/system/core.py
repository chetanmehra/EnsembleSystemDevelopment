'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from pandas import DateOffset, Panel, DataFrame, Series

from system.interfaces import IndexerFactory
from data_types.trades import TradeCollection
from data_types.positions import Position, Returns


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
        returns.plot(start = start, color = color, **kwargs)
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
        self.summary = DataFrame(0, index = self.share_holdings.index, columns = ["Cash", "Holdings", "Total"])
        self.summary["Cash"] = starting_cash
        self.costs = DataFrame(0, index = self.share_holdings.index, columns = ["Commissions", "Slippage", "Total"])
        self.trade_size = (2000, 3500) # Min and max by dollar size
        self.conditions = [MinimimumPositionSize()]
        self.transaction_checks = [TransactionCostThreshold(0.02)]

    def run(self):
        """
        Runs the portfolio for the given strategy and starting conditions.
        This calculates the resulting portfolio based on positions sizes.
        """
        trading_days = self.share_holdings.index
        for date in trading_days:
            target_txns = self.get_target_transactions(date)
            selected_txns = self.check_transations(target_txn)
            # TODO need to make sure that exit transactions haven't been removed accidentally.
            # This may occur if the size of the position has reduced to a point where the cost to
            # exit would be higher than the threshold allowed. However in this case we still want
            # to exit the position.
            self.apply(selected_txns, date)
        self.dollar_holdings = self.share_holdings * self.strategy.get_trade_prices()
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]
        #TODO Use a Positions object to calculate trades.

    def get_target_transactions(self, date):
        """
        In essence the target transactions are calculated from the difference
        between the current positions, and desired positions on a given day.
        This also needs to convert the generic position sizing into the size
        relevant for the portfolio, as well as removing positions which don't 
        fit within the constraints of the portfolio size (e.g. leverage constraints).
        """

        # 1. Get the target positions
        # 2. Get the current positions
        #       Note, may need to hold the generic position size if we want to 
        #       separate raw position changes from rebalancing position changes.
        # 3. Target changes = target positions - current positions
        # 4. Convert the target changes to number of shares.
        pass

    def check_transactions(self, target_transactions):
        """
        This runs through a series of checks, removing any transactions that don't
        meet the conditions (e.g. if cost would be too high).
        """
        for check in self.transaction_checks:
            target_transactions = check(target_transactions)
        return target_transactions

    def apply(self, transactions, date):
        self.cash[date:] -= transactions.total_cost
        self.costs.loc[date:, "Commissions"] += transactions.total_commissions
        self.costs.loc[date:, "Slippage"] += transactions.total_slippage
        self.share_holdings[date:] += transactions.num_shares

    def apply_trades(self):
        """
        Given a TradeCollection from a Strategy, determine the selected trades
        and resulting portfolio positions.
        """
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
                entry_txns = Transactions(num_shares, trade_prices)
                self.apply(entry_txns, trade.entry)
                trade_prices = self.strategy.get_trade_prices().loc[trade.exit]
                exit_txns = Transactions(-1 * num_shares, trade_prices)
                self.apply(exit_txns, trade.exit)
        self.trades = TradeCollection(executed_trades)
        self.dollar_holdings = self.share_holdings * self.strategy.get_trade_prices()
        self.dollar_holdings = self.dollar_holdings.fillna(method = 'ffill')
        self.summary["Holdings"] = self.dollar_holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total
        self.costs["Total"] = self.costs["Commissions"] + self.costs["Slippage"]

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
        return self.summary["Holdings"]

    @property
    def value(self):
        '''
        Returns the series of total dollar value of the portfolio
        '''
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
    def __init__(self, num_shares, prices):
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
        cost_ratio = (transactions.commissions + transactions.slippage) / transactions.dollar_positions
        costly_tickers = cost_ratio[cost_ratio > self.cost_threshold].index
        transactions.remove(transactions.tickers)
        return transactions


