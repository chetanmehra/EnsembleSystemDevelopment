
from pandas import Series, DataFrame

class Portfolio(object):
    '''
    The portfolio class manages cash and stock positions, as well as rules 
    regarding the size of positions to hold and rebalancing.
    '''
    def __init__(self, strategy, starting_cash):
        self.strategy = strategy
        self.positions = self.strategy.get_empty_dataframe(0) # Number of shares
        self.cash = Series(starting_cash, self.positions.index)
        self.min_size = 2000 # Min dollar size
        self.transaction_cost = 11 # Each way

    def apply_trades(self):
        trade_df = self.strategy.trades.as_dataframe().sort_values(by = 'entry')
        for i in range(len(trade_df)):
            trade = trade_df.iloc[i]
            if self.cash[trade.entry] > self.min_size:
                num_shares = int(self.min_size / trade.entry_price)
                cost = (num_shares * trade.entry_price) + self.transaction_cost
                self.cash[trade.entry:] -= cost
                self.positions[trade.ticker][trade.entry:trade.exit] = num_shares
                sale_proceeds = (num_shares * trade.exit_price) - self.transaction_cost
                self.cash[trade.exit:] += sale_proceeds
        self.holdings = self.positions * self.strategy.get_trade_prices()

    def holdings_total(self):
        return self.holdings.sum(axis = 1)

    def plot_result(self):
        plot_df = DataFrame()
        plot_df["Cash"] = self.cash
        plot_df["Holdings"] = self.holdings_total()
        plot_df["Total"] = self.cash + self.holdings_total()
        plot_df.plot()



    
