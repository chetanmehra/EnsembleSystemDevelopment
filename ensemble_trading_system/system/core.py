'''
Created on 21 Dec 2014

@author: Mark
'''
import matplotlib.pyplot as plt
from pandas import DateOffset, Panel, DataFrame, Series

from system.interfaces import Indexer
from data_types.trades import Trade, TradeCollection, create_trades
from system.metrics import Drawdowns


class Strategy:
    '''
    Strategy defines the base interface for Strategy objects
    '''
    def __init__(self, trade_timing, ind_timing):
        self.indexer = Indexer(trade_timing, ind_timing)
        self.required_fields = ["market", "signal_generator", "position_rules"]
        self.name = None
        self.signal = None
        self.positions = None
        self.signal_generator = None
        self.position_rules = None
        self.filters = []

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
        self.trades = create_trades(self.positions.data, self)

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
        if self.trade_timing[0] == "O":
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
        return self.market.returns(self.indexer)

    # TODO Strategy returns should be an AverageReturns object, not AggregateReturns.
    
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
# TODO Portfolio returns should be an AggregateReturns object

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
        self.positions = self.strategy.get_empty_dataframe(0) # Number of shares
        self.holdings = self.strategy.get_empty_dataframe(0) # Dollar value of positions
        self.summary = DataFrame(0, index = self.positions.index, columns = ["Cash", "Holdings", "Total"])
        self.summary["Cash"] = starting_cash
        self.trade_size = (2000, 3500) # Min and max by dollar size
        self.transaction_cost = 11 # Each way
        self.conditions = [minimum_position_size]

    def apply_trades(self):
        '''
        Creates the portfolio positions and totals based on the strategy trades and position rules.
        '''
        trade_df = self.strategy.trades.as_dataframe().sort_values(by = 'entry')
        executed_trades = []
        for i in range(len(trade_df)): # Note index is unordered due to sort so need to create new index range for loop.
            trade = trade_df.iloc[i]
            trade_num = trade_df.index[i]
            executed_trades.append(self.strategy.trades[trade_num])
            if all(self.conditions_met_for(trade)):
                num_shares = int(min(self.cash[trade.entry], max(self.trade_size)) / trade.entry_price)
                cost = (num_shares * trade.entry_price) + self.transaction_cost
                self.cash[trade.entry:] -= cost
                self.positions[trade.ticker][trade.entry:(trade.exit - DateOffset(1))] = num_shares
                sale_proceeds = (num_shares * trade.exit_price) - self.transaction_cost
                self.cash[trade.exit:] += sale_proceeds
        self.trades = TradeCollection(executed_trades)
        self.holdings = self.positions * self.strategy.get_trade_prices()
        self.summary["Holdings"] = self.holdings.sum(axis = 1)
        self.summary["Total"] = self.cash + self.holdings_total

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
    def returns(self):
        '''
        Returns the daily returns for the portfolio
        '''
        returns = self.value / self.value.shift(1) - 1
        returns[0] = 0
        return returns

    @property
    def cumulative_returns(self):
        '''
        Returns the cumulative returns series for the portfolio
        '''
        return self.value / self.value[0] - 1

    @property
    def drawdowns(self):
        '''
        Returns the calculated drawdowns and highwater series for the portfolio returns.
        '''
        return Drawdowns(self.cumulative_returns)

    def plot_result(self, start = None, dd_ylim = None, rets_ylim = None):
        '''
        Plots the portfolio returns and drawdowns vs the market.
        '''
        f, axarr = plt.subplots(2, 1, sharex = True)
        axarr[0].set_ylabel('Return')
        axarr[1].set_ylabel('Drawdown')
        axarr[1].set_xlabel('Days in trade')
        self.cumulative_returns[start:].plot(ax = axarr[0], ylim = rets_ylim)
        dd = self.drawdowns
        dd.Highwater[start:].plot(ax = axarr[0], color = 'red')
        dd.Drawdown[start:].plot(ax = axarr[1], ylim = dd_ylim)
        mkt_returns = self.strategy.market_returns
        mkt_dd = Drawdowns(mkt_returns.cumulative()).Drawdown
        mkt_returns.plot(start = start, ax = axarr[0], color = 'black')
        mkt_dd[start:].plot(ax = axarr[1], color = 'black')
        return axarr


# Accepts a trade if the current cash position is greater than the minimum trade size.
def minimum_position_size(portfolio, trade):
    '''
    Provides a trade condition where the cash at the time of entry must be greater
    than the minimum allowable portfolio size.
    '''
    return portfolio.cash[trade.entry] > min(portfolio.trade_size)
