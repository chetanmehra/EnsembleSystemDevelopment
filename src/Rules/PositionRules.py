
from System.Strategy import PositionRuleElement
from System.Position import Position

class PositionFromDiscreteSignal(PositionRuleElement):
    '''
    Creates positions sized based on the signal level.
    The position size is specified per level using keyword args. 
    If no size is specified it is assumed to be zero.
    '''
    def __init__(self, **kwargs):
        self.level_sizes = kwargs

    def execute(self, strategy):
        pos_data = strategy.getEmptyDataFrame(fill_data = 0)
        signal = strategy.signal.at("entry")
        for level, position_size in self.level_sizes.items():
            pos_data[signal.data == level] = position_size
        return Position(pos_data)


class OptimalFSign(PositionRuleElement):
    '''
    Returns positions sized according to the sign of the opt F calc (i.e. 1 or -1).
    Requires a forecaster.
    '''
    def __init__(self, forecaster):
        self.forecast = forecaster

    def execute(self, strategy):
        forecasts = self.forecast(strategy)
        optimal_size = forecasts.optF()
        positions = Position(sign(optimal_size))
        positions.forecasts = forecasts
        return positions

    @property
    def name(self):
        return 'Default positions'


class OptimalPositions(PositionRuleElement):

    def __init__(self, forecaster):
        self.forecast = forecaster

    def execute(self, strategy):
        forecasts = self.forecast(strategy)
        optimal_size = forecasts.optF()
        positions = Position(optimal_size)
        positions.forecasts = forecasts
        return positions

    @property
    def name(self):
        return 'Optimal F positions'


class SingleLargestF(PositionRuleElement):

    def __init__(self, forecaster):
        self.forecast = forecaster
    
    def execute(self, strategy):
        forecasts = self.forecast(strategy)
        optimal_size = forecasts.optF()
        result = deepcopy(optimal_size)
        result[:] = 0
        maximum_locations = optimal_size.abs().idxmax(axis = 1)
        directions = sign(optimal_size)
        
        for row, col in enumerate(maximum_locations):
            if col not in result.columns:
                pass
            else:
                result[col][row] = directions[col][row] 
        
        positions = Position(result)
        positions.forecasts = forecasts
        return positions
    
    @property
    def name(self):
        return 'Single Largest F'
    
class HighestRankedFs(PositionRuleElement):
    '''
    Provides equal weighted positions of the 'n' highest ranked opt F forecasts.
    '''
    def __init__(self, forecaster, num_positions):
        self.forecast = forecaster
        self.num_positions = num_positions
        self.name = '{} Highest Ranked Fs'.format(num_positions)
        
    def execute(self, strategy):
        forecasts = self.forecast(strategy)
        optimal_size = forecasts.optF()
        result = deepcopy(optimal_size)
        result[:] = 0
        ranks = abs(optimal_size).rank(axis = 1, ascending = False)
        result[ranks <= self.num_positions] = 1
        result *= sign(optimal_size)
        result /= self.num_positions
        positions = Position(result)
        positions.forecasts = forecasts
        return positions

  
