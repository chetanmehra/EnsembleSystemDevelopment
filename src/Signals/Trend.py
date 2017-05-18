
from System.Strategy import MeasureElement
from System.Trade import Trade, TradeCollection
from pandas import Panel

class LagDecorator(object):

    def __init__(self, measure, lag):
        self.measure = measure
        self.lag = lag

    def __call__(self, strategy):
        base_trades = self.measure(strategy)
        entry_prices = strategy.get_entry_prices()
        exit_prices = strategy.get_exit_prices()
        new_trades = []
        for trade in base_trades.as_list():
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

    def update_param(self, new_params):
        self.measure.update_param(new_params)


class Crossover(object):

    def __init__(self, slow, fast):
        self.fast = fast
        self.slow = slow
        self.measures = None

    @property
    def name(self):
        return "x".join([self.fast.name, self.slow.name])

    def __call__(self, strategy):
        prices = strategy.get_indicator_prices()
        fast_ema = self.fast(prices)
        slow_ema = self.slow(prices)
        self.data = fast_ema > slow_ema
        self.measures = Panel.from_dict({'Fast':fast_ema, 'Slow':slow_ema})
        return self.create_trades(strategy.get_entry_prices(), strategy.get_exit_prices())


    def create_trades(self, entry_prices, exit_prices):
        trades = []
        flags = self.data - self.data.shift(1)
        flags.ix[0] = 0
        flags.ix[0][self.data.ix[0]] = 1
        for ticker in flags:
            ticker_flags = flags[ticker]
            entries = ticker_flags.index[ticker_flags > 0]
            i = 0
            while i < len(entries):
                entry = entries[i]
                i += 1
                if i < len(entries):
                    next = entries[i]
                else:
                    next = None
                exit = ticker_flags[entry:next].index[ticker_flags[entry:next] < 0]
                if len(exit) == 0:
                    exit = ticker_flags.index[-1]
                else:
                    exit = exit[0]
                trades.append(Trade(ticker, entry, exit, entry_prices[ticker], exit_prices[ticker]))
        return TradeCollection(trades)

    
    def update_param(self, new_params):
        self.slow.update_param(new_params[0])
        self.fast.update_param(new_params[1])

    def plot_measures(self, ticker, start = None, end = None, ax = None):
        self.measures.minor_xs(ticker)[start:end].plot(ax = ax)

        

