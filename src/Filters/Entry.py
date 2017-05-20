
from System.Filter import FilterInterface
from System.Trade import Trade, TradeCollection


class EntryLagFilter(FilterInterface):

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, strategy):
        entry_prices = strategy.get_entry_prices()
        exit_prices = strategy.get_exit_prices()
        new_trades = []
        for trade in strategy.trades.as_list():
            if trade.duration <= self.lag:
                continue
            new_possible_entries = trade.normalised.index[trade.normalised > 0]
            new_possible_entries = new_possible_entries[new_possible_entries > self.lag]
            if len(new_possible_entries) == 0:
                continue
            new_entry_lag = min(new_possible_entries) + 1
            if new_entry_lag >= trade.duration:
                continue
            new_entry = entry_prices[trade.entry:].index[new_entry_lag]
            new_trades.append(Trade(trade.ticker, new_entry, trade.exit, entry_prices[trade.ticker], exit_prices[trade.ticker]))
        return TradeCollection(new_trades)

    def plot(self, ticker, start, end, ax):
        pass
