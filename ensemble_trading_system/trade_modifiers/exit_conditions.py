
class ExitCondition:

    def get_limit_hits(self, trade):
        raise NotImplementedError("Exit condition must impelement get_limit_hits")

    def __call__(self, trade):
        limit_hits = self.get_limit_hits(trade)
        if len(limit_hits):
            return trade.revise_exit(min(limit_hits))
        else:
            return trade

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

class PositionBasedStop(StopLoss):
    '''
    The PositionBasedStop varies its stop level based on the position size
    of the trade it is applied to. This accounts for adjustments such as 
    volatility sizing applied to the position, where you would have a stop
    placement inversely proportional to the position (proportional to vol).
    In this case stop_level refers to the base stop percentage for a position 
    size of one.
    '''
    def __init___(self, stop_level):
        self.base_stop = -1 * abs(stop_level)
        self.stop = self.base_stop

    def get_limit_hits(self, trade):
        '''
        This stop is not designed for strategies with daily variable
        position sizes (i.e. where trade.position_size is a series)
        '''
        self.stop = self.base_stop / trade.position_size
        super().get_limit_hits(trade)

