
from copy import deepcopy
from numpy import sign

from rules import PositionRuleElement, Position


class CarterPositions(PositionRuleElement):
    """
    Calculates positions based on the normalised forecast (-20 to 20), 
    and the current standard deviation of the instrument.
    """
    def __init__(self, volatility_method, target_volatility, long_only = True):
        self.volatility_method = volatility_method
        self.target_vol = target_volatility
        self.long_only = long_only

    def execute(self, strategy):
        forecasts = strategy.signal.at("entry")
        current_volatility = self.volatility_method(strategy.get_indicator_prices())
        base_position = 10 * self.target_vol / current_volatility # This is the size we should hold for a forecast of 10
        positions = forecasts.data / base_position
        if self.long_only:
            positions[positions < 0] = 0
        return Position(positions)


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

  
