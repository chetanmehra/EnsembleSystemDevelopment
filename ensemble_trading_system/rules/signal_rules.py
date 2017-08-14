
from rules import PositionRuleElement, Position

class PositionFromDiscreteSignal(PositionRuleElement):
    '''
    Creates positions sized based on the signal level.
    The position size is specified per level using keyword args. 
    If no size is specified it is assumed to be zero.
    '''
    def __init__(self, **kwargs):
        self.level_sizes = kwargs
        self.name = ", ".join(["{} [{}]".format(key, value) for key, value in self.level_sizes.items()])

    def execute(self, strategy):
        pos_data = strategy.get_empty_dataframe(fill_data = 0)
        signal = strategy.signal.at(strategy.entry)
        for level, position_size in self.level_sizes.items():
            pos_data[signal.data == level] = position_size
        return Position(pos_data)
