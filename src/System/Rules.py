# Rules create positions from a trade set
from pandas import DataFrame
from System.Trade import TradeCollection
from System.Position import Position

class EqualSizing():

    def __init__(self, num_positions, pos_size = None):
        self.num_positions = num_positions
        if pos_size is None:
            pos_size = 1 / num_positions
        self.pos_size = pos_size

    def __call__(self, strat):
        trades_taken = []
        df = strat.trades.as_dataframe().sort_values(by = 'entry')
        open_positions = DataFrame(columns = df.columns)

        for t in df.index:
            possible_trade = strat.trades[t]
            open_positions = open_positions[open_positions.exit > possible_trade.entry]

            if len(open_positions) < self.num_positions:
                open_positions = open_positions.append(df.loc[t])
                trades_taken.append(possible_trade)

        pos_data = strat.get_empty_dataframe()
        pos_data[:] = 0
        for T in trades_taken:
            pos_data.loc[T.entry:T.exit, T.ticker] = self.pos_size

        positions = Position(pos_data)
        positions.trades = TradeCollection(trades_taken)

        return positions




