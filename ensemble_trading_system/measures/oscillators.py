
import pandas as pd

class BollingerBands(SignalElement):
    
    def __init__(self, moving_average, vol_method, bands):
        '''
        Accepts moving average, and volatility calculation methods.
        Uses these along with the provided bands parameter to provide
        the measure.
        bands should be a scalar multiple (e.g. 1.5 or 2) to be applied
        to the volatility.
        '''
        self.moving_average = moving_average
        self.vol_method = vol_method
        self.bands = bands

    @property
    def name(self):
        return "Bollinger.{0}.{1}x{2}".format(self.moving_average.name, self.bands, self.vol_method.name)

    def update_param(self, new_params):
        self.vol_period = new_params[0]
        self.bands = new_params[1]

    def __call__(self, prices):
        '''
        Returns a panel with the moving average, and
        upper and lower volatility bands.
        '''
        moving_average = self.moving_average(prices)
        volatility = self.vol_method(prices)
        upper = moving_average + self.bands * volatility
        lower = moving_average - self.bands * volatility
        return pd.Panel.from_dict({"average" : moving_average, 
                                   "upper" : upper, 
                                   "lower" : lower})



