
class TradeWeighting:

    def __init__(self, weights):
        self.values = weights

    def __call__(self, trade):
        weight = self.get(trade.entry, trade.ticker)
        if weight is not None:
            trade.position_size = weight
        return trade

    def get(self, date, ticker):
        try:
            value = self.values.loc[date, ticker]
        except KeyError:
            value = None
        return value