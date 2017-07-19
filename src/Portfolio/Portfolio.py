
from pandas import DataFrame, DateOffset
import matplotlib.pyplot as plt

from System.Trade import TradeCollection
from PerformanceAnalysis.Metrics import Drawdowns

class Portfolio(object):
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
        return self.summary["Cash"]

    @property
    def holdings_total(self):
        return self.summary["Holdings"]

    @property
    def value(self):
        return self.summary["Total"]

    @property
    def returns(self):
        returns = self.value / self.value.shift(1) - 1
        returns[0] = 0
        return returns

    @property
    def cumulative_returns(self):
        return self.value / self.value[0] - 1

    @property
    def drawdowns(self):
        return Drawdowns(self.cumulative_returns)

    def plot_result(self, start = None, dd_ylim = (-0.25, 0), rets_ylim = (-0.2, 1)):
        f, axarr = plt.subplots(2, 1, sharex = True)
        axarr[0].set_ylabel('Return')
        axarr[1].set_ylabel('Drawdown')
        axarr[1].set_xlabel('Days in trade')
        self.cumulative_returns[start:].plot(ax = axarr[0], ylim = rets_ylim)
        dd = self.drawdowns
        dd.Highwater[start:].plot(ax = axarr[0], color = 'red')
        dd.Drawdown[start:].plot(ax = axarr[1], ylim = dd_ylim)
        mkt_returns = self.strategy.market_returns.cumulative()
        mkt_dd = Drawdowns(mkt_returns).Drawdown
        mkt_returns[start:].plot(ax = axarr[0], color = 'black')
        mkt_dd[start:].plot(ax = axarr[1], color = 'black')
        return axarr


# Accepts a trade if the current cash position is greater than the minimum trade size.
def minimum_position_size(portfolio, trade):
    return portfolio.cash[trade.entry] > min(portfolio.trade_size)

    
