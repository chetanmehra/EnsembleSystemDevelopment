
from System.Strategy import PositionSelectionElement
from System.Position import Position

class PositionFromDiscreteSignal(PositionSelectionElement):
    '''
    Creates positions sized based on the signal level.
    The position size is specified per level using keyword args. 
    If no size is specified it is assumed to be zero.
    '''
    def __init__(self, **kwargs):
        self.level_sizes = kwargs

    def execute(self, strategy):
        pos_data = strategy.get_empty_dataframe()
        pos_data[:] = 0
        # TODO signal data needs to be lagged
        # strategy.signal.at("entry")
        signal = strategy.signal.at("entry")
        for level, position_size in self.level_sizes.items():
            pos_data[signal.data == level] = position_size
        return Position(pos_data)
