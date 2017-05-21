# Rules create positions from a trade set
from pandas import DataFrame
from System.Trade import TradeCollection
from System.Position import createPositionFromTrades

class EqualSizing():

    def __init__(self, num_positions, pos_size = None):
        self.num_positions = num_positions
        if pos_size is None:
            pos_size = 1 / num_positions
        self.pos_size = pos_size

    def __call__(self, strategy):
        trades_taken = []
        df = strategy.trades.as_dataframe().sort_values(by = 'entry')
        open_positions = DataFrame(columns = df.columns)

        for t in df.index:
            possible_trade = strategy.trades[t]
            open_positions = open_positions[open_positions.exit > possible_trade.entry]

            if len(open_positions) < self.num_positions:
                open_positions = open_positions.append(df.loc[t])
                trades_taken.append(possible_trade)

        positions = createPositionFromTrades(trades_taken, strategy, self.pos_size)
        positions.trades = TradeCollection(trades_taken)

        return positions

