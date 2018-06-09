
from pandas import Series

class ExitCondition:

    def get_limit_hits(self, trade):
        raise NotImplementedError("Exit condition must impelement get_limit_hits")

    def __call__(self, trade):
        limit_hits = self.get_limit_hits(trade)
        if len(limit_hits):
            return min(limit_hits)
        else:
            return None

class TrailingStop(ExitCondition):

    def __init__(self, stop_level):
        self.stop = -1 * abs(stop_level)

    def get_limit_hits(self, trade):
        drawdowns = trade.drawdowns().Drawdown
        return drawdowns.index[drawdowns <= self.stop]

class StopLoss(ExitCondition):

    def __init__(self, stop_level):
        self.stop = -1 * abs(stop_level)

    def get_limit_hits(self, trade):
        returns = trade.cumulative
        return returns.index[returns <= self.stop]

class ReturnTriggeredTrailingStop(ExitCondition):

    def __init__(self, stop_level, return_level):
        self.stop = -1 * abs(stop_level)
        self.return_level = return_level

    def get_limit_hits(self, trade):
        dd = trade.drawdowns()
        drawdowns = dd.Drawdown
        highwater = dd.Highwater
        return highwater.index[(highwater >= self.return_level) & (drawdowns <= self.stop)]

