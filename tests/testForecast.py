'''
Created on 20 Dec 2014

@author: Mark
'''
import unittest
from mock import Mock
from System.Forecast import BlockForecaster, Forecast, BlockMeanReturnsForecaster
from pandas import Series, DataFrame
from numpy import mean, std
from math import isnan
from tests.TestHelpers import buildNumericPanel, buildTextDataFrame, buildNumericDataFrame
from System.Indicator import Signal
from System.Strategy import ModelStrategy
from System.Position import AverageReturns


class TestForecastInitialisation(unittest.TestCase):
    
    def setUp(self):
        tickers = ["ASX", "BHP"]
        columns = ["Forecast", "A", "B"]
        self.mean = buildNumericPanel(tickers, columns, 5)
        self.sd = buildNumericPanel(tickers, columns, 5)
        
    def testForecastRequiresMeanAndSdOnInitialise(self):
        self.assertRaises(TypeError, Forecast)
        Forecast(self.mean, self.sd)


class TestForecastInterface(unittest.TestCase):
    
    def setUp(self):
        tickers = ["ASX", "BHP"]
        columns = ["Forecast", "A", "B"]
        self.mean = buildNumericPanel(columns, tickers, 5)
        self.sd = buildNumericPanel(columns, tickers, 5)
        self.forecasts = Forecast(self.mean, self.sd)
        
    def testForecastCalcsOptF(self):
        expected_optF = self.mean["Forecast"] / (self.sd["Forecast"]**2)
        self.assertTrue(expected_optF.equals(self.forecasts.optF()))
        
    def testForecastOptFwithZeroSD(self):
        self.sd["Forecast"]["ASX"][:] = 0
        result = self.forecasts.optF()
        self.assertEqual(result["ASX"][0], 0.0)
        self.assertTrue(all(result["ASX"] == 0))
        
    def testForecastMeanAndSDReturnsDataFrame(self):
        self.assertTrue(self.forecasts.mean.equals(self.mean["Forecast"]))
        self.assertTrue(self.forecasts.sd.equals(self.sd["Forecast"]))
        

class TestBlockForecasterConstruction(unittest.TestCase):

    def setUp(self):
        self.strategy = Mock(spec = ModelStrategy)
        self.model = BlockForecaster(window = 4)
        self.ticker = "ASX"
        self.inds = Series([str(B) for B in [True, True, False, True, True, False, False, False]])
        self.rtns = Series([0.1, 0.15, 0.2, 0.3, 0.15, 0.22, 0.18, 0.16])
        self.lagged_indicator = Signal(DataFrame({self.ticker:self.inds}))
        self.strategy.lagged_indicator = self.lagged_indicator
        self.returns = DataFrame({self.ticker:self.rtns})
        self.strategy.market_returns = self.returns
        self.strategy.forecasts = None
        
    def testModelReturnsForecastObject(self):
        self.assertIsInstance(self.model(self.strategy), Forecast)
        
    def testModelProducesResultsForAllData(self):
        tickers = ["ASX", "BHP", "CBA"]
        indicator_data = buildTextDataFrame(tickers, 10)
        self.strategy.lagged_indicator = Signal(indicator_data)
        self.strategy.market_returns = buildNumericDataFrame(tickers, 10)
        forecast = self.model(self.strategy)
        self.assertEqual(indicator_data.shape, forecast.mean.shape)
        self.assertEqual(indicator_data.shape, forecast.sd.shape)
        self.assertTrue(all(indicator_data.index == forecast.mean.index))
        self.assertTrue(all(indicator_data.index == forecast.sd.index))
        self.assertEqual(tickers, list(forecast.mean.columns))
        self.assertEqual(tickers, list(forecast.sd.columns))

    def testModelProducesSeriesForEachLevel(self):
        mean_sd = self.model.forecast_mean_sd(self.lagged_indicator, self.returns)
        for result in mean_sd:
            self.assertEqual(list(result.items), ["Forecast", "False", "True"])
        
    def testModelCalcsRollMeanAndStdDev(self):
        mu, sd = self.model.forecast_mean_sd(self.lagged_indicator, self.returns)
        
        self.assertTrue(isnan(mu["True"]["ASX"][2]))
        self.assertEqual(mu["True"]["ASX"][3], mean(self.rtns[self.inds.values[0:4] == "True"]))
        self.assertEqual(mu["True"]["ASX"][7], 
                         mean(self.rtns[4:][self.inds.values[4:] == "True"]))
        self.assertEqual(mu["False"]["ASX"][7], 
                         mean(self.rtns[4:][self.inds.values[4:] == "False"]))
        
        self.assertTrue(isnan(sd["True"]["ASX"][2]))
        self.assertEqual(sd["True"]["ASX"][3], std(self.rtns[self.inds.values[0:4] == "True"]))
        self.assertEqual(sd["True"]["ASX"][7], std(self.rtns[4:][self.inds.values[4:] == "True"]))
        self.assertEqual(sd["False"]["ASX"][7], std(self.rtns[4:][self.inds.values[4:] == "False"]))
        
    def testModelAssignsForecastFromLevel(self):
        mu, sd = self.model.forecast_mean_sd(self.lagged_indicator, self.returns)
        
        self.assertEqual(mu["Forecast"]["ASX"][4], mu["True"]["ASX"][4])
        self.assertEqual(mu["Forecast"]["ASX"][5], mu["False"]["ASX"][5])
        self.assertEqual(mu["Forecast"]["ASX"][6], mu["False"]["ASX"][6])
        
        self.assertEqual(sd["Forecast"]["ASX"][4], sd["True"]["ASX"][4])
        self.assertEqual(sd["Forecast"]["ASX"][5], sd["False"]["ASX"][5])
        self.assertEqual(sd["Forecast"]["ASX"][6], sd["False"]["ASX"][6])
        
    def testExecuteHandlesNoValuesInWindow(self):
        self.strategy.lagged_indicator["ASX"][3:5] = "False"
        self.inds = self.strategy.lagged_indicator["ASX"]
        result = self.model(self.strategy)
        self.assertEqual(result._mean["True"]["ASX"][3], 
                         mean(self.rtns[self.inds.values[0:4] == "True"]))
        self.assertTrue(isnan(result._mean["True"]["ASX"][5]))
        self.assertTrue(isnan(result._mean["True"]["ASX"][6]))
        self.assertTrue(isnan(result._mean["True"]["ASX"][7]))
        
    def testBlockForecasterUpdatesParameters(self):
        self.assertEqual(self.model.window, 4)
        self.model.update_param(10)
        self.assertEqual(self.model.window, 10)


class TestBlockMeanReturnsForecaster(unittest.TestCase):
    
    def setUp(self):
        self.strategy = Mock(spec = ModelStrategy)
        self.model = BlockMeanReturnsForecaster(window = 4)
        self.tickers = ["ASX", "BHP", "CBA"]
        self.returns = AverageReturns(buildNumericDataFrame(self.tickers, 10))
        self.strategy.market_returns = self.returns
        self.strategy.forecasts = None
        
    def  testModelReturnsForecastObject(self):
        self.assertIsInstance(self.model(self.strategy), Forecast)
    
    def testModelProducesResultsForAllData(self):
        forecast = self.model(self.strategy)
        self.assertEqual(self.returns.data.shape, forecast.mean.shape)
        self.assertEqual(self.returns.data.shape, forecast.sd.shape)
        self.assertTrue(all(self.returns.data.index == forecast.mean.index))
        self.assertTrue(all(self.returns.data.index == forecast.sd.index))
        self.assertEqual(self.tickers, list(forecast.mean.columns))
        self.assertEqual(self.tickers, list(forecast.sd.columns))
         
           

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()