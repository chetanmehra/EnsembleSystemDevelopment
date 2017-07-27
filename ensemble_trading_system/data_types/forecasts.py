'''
Created on 20 Dec 2014

@author: Mark
'''


class Forecast:
    '''
    A Forecast element supports position calculations for certain position rules.
    Given the mean and standard deviation forecasts over time, it calculates the optimal F.
    '''
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
    
    @property
    def mean(self):
        return self._mean["Forecast"]
    
    @mean.setter
    def mean(self, data):
        self._mean = data
        
    @property
    def sd(self):
        return self._sd["Forecast"]
    
    @sd.setter
    def sd(self, data):
        self._sd = data
    
    def optF(self):
        optimal_fraction = self.mean / self.sd ** 2
        optimal_fraction[isinf(optimal_fraction)] = 0 
        return optimal_fraction
    
    
class MeanForecastWeighting:
    
    def __call__(self, forecasts):
        items = list(bounds(len(forecasts)))
        means = [forecast.mean for forecast in forecasts]
        means = zip(items, means)
        stds = [forecast.sd for forecast in forecasts]
        stds = zip(items, stds)
        means = Panel({item:frame for item, frame in means})
        stds = Panel({item:frame for item, frame in stds})
        return MeanForecast(means.mean(axis = "items"), stds.mean(axis = "items"))
    
    
class MeanForecast(Forecast):
    
    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, data):
        self._mean = data
    
    @property 
    def sd(self):
        return self._sd
    
    @sd.setter
    def sd(self, data):
        self._sd = data

     


    
    
