'''
Created on 28 Jan 2015

@author: Mark
'''
import unittest
from System.Position import Position, SingleLargestF, Returns, HighestRankedFs
from mock import Mock
from copy import deepcopy
from numpy import sign, NaN
from math import log
from tests.TestHelpers import buildNumericDataFrame
from System.Forecast import Forecast
from System.Strategy import ModelStrategy


class TestPositionHolder(unittest.TestCase):

    def setUp(self):
        self.tickers = ['ASX', 'BHP', 'CBA']
        self.position_data = buildNumericDataFrame(self.tickers, 10)
        self.position = Position(self.position_data)

    def testPositionExpectsDataFrame(self):
        self.assertRaises(TypeError, Position, [1, 2])
        self.assertRaises(TypeError, Position, self.tickers)
        
    def testPositionStoresPositionDataOnCreation(self):
        self.assertIs(self.position.data, self.position_data)
        
    def testPositionsReturnsLongOnly(self):
        result = self.position.long_only()
        self.assertEqual(min(result.data.min()), 0)
        self.assertFalse(result.data.equals(self.position.data))
        
    def testPositionsReturnsShortOnly(self):
        result = self.position.short_only()
        self.assertEqual(max(result.data.max()), 0)
        self.assertFalse(result.data.equals(self.position.data))


class TestReturnsObject(unittest.TestCase):
    
    def setUp(self):
        self.data = buildNumericDataFrame(["ASX", "BHP", "CBA"], 20) / 10
        self.returns = Returns(self.data)
        
    
    def testReturnsProvidesCumulativeMeanLog(self):
        expected = self.data.sum(axis = 1)
        expected = (expected + 1).apply(log)
        expected = expected.cumsum()
        self.assertTrue(self.returns.cumulative(collapse_fun = "sum").equals(expected))
        
    def testReturnsProvidesCumulativeSumLog(self):
        expected = self.data.mean(axis = 1)
        expected = (expected + 1).apply(log)
        expected = expected.cumsum()
        self.assertTrue(self.returns.cumulative(collapse_fun = "mean").equals(expected))
        
    def testReturnsCollapseColumns(self):
        self.assertRaisesRegex(TypeError, "function must be one of: mean, sum", 
                               self.returns.collapse_by, "X")
        self.assertTrue(self.returns.collapse_by("mean").equals(self.data.mean(axis = 1)))
        self.assertTrue(self.returns.collapse_by("sum").equals(self.data.sum(axis = 1)))
        
    def testReturnsColumnsLabels(self):
        self.assertTrue(all(self.returns.columns == self.data.columns))
        
    def testReturnsCombinesTogether(self):
        data2 = buildNumericDataFrame(['a', 'b'], len(self.data))
        expected_data = self.returns.data
        expected_data[data2.columns] = data2
        returns2 = Returns(data2)
        self.returns.append(returns2)
        self.assertTrue(self.returns.data.equals(expected_data))
        
        


class TestPositionModel(unittest.TestCase):

    def setUp(self):
        self.tickers = ['ASX', 'BHP', 'CBA']
        self.forecast_data = buildNumericDataFrame(self.tickers, 10)
        self.forecasts = Mock(spec = Forecast)
        self.forecasts.optF.return_value = self.forecast_data
        self.strategy = ModelStrategy("OO", "C")
        self.strategy.forecasts = self.forecasts
        self.model = SingleLargestF()
        self.model_results = self.model(self.strategy)
        
    def testPositionModelReturnsPosition(self):
        self.assertIsInstance(self.model_results, Position)
    
    def testPositionModelReturnsSameSizeAsForecast(self):
        self.assertEqual(self.forecast_data.shape, self.model_results.data.shape)
        
    def testPositionModelUsesOptimalF(self):
        self.forecasts.optF.assert_called_once_with()
        
    def testPositionModelFindsMaxSizeAtEachTime(self):
        expected = deepcopy(self.forecast_data)
        expected[:] = 0
        for i in self.forecast_data.index:
            values = list(self.forecast_data.loc[i])
            magnitudes = [abs(value) for value in values]
            position = [sign(value) if abs(value) == max(magnitudes) else 0 for value in values]
            expected.loc[i] = position
        self.assertTrue(expected.equals(self.model_results.data))
        
    def testPositionModelHandlesMissingValues(self):
        self.forecast_data[:] = NaN
        expected = deepcopy(self.forecast_data)
        expected[:] = 0
        self.assertTrue(expected.equals(self.model(self.strategy).data))    


class TestHighestRankedFsSelector(unittest.TestCase):
    
    def setUp(self):
        self.tickers = ['ASX', 'BHP', 'CBA', 'MOC', 'QBE']
        self.forecast_data = buildNumericDataFrame(self.tickers, 10)
        self.forecasts = Mock(spec = Forecast)
        self.forecasts.optF.return_value = self.forecast_data
        self.strategy = ModelStrategy("OO", "C")
        self.strategy.forecasts = self.forecasts
        self.num_positions = 3
        self.select_positions = HighestRankedFs(self.num_positions)
        self.positions = self.select_positions(self.strategy)
        
    def testReturnsPositionObject(self):
        self.assertIsInstance(self.positions, Position)
        
    def testOptFCalledOnExecute(self):
        self.forecasts.optF.assert_called_once_with()
        
    def testNumberOfPositionsSelected(self):
        num_positions_rows = (self.positions.data != 0).sum(axis = 1)
        self.assertTrue(all(num_positions_rows == self.num_positions))
        
    def testPositionsSumToOneAbsolute(self):
        position_total_magnitude = abs(self.positions.data).sum(axis = 1)
        self.assertTrue(all(position_total_magnitude == 1.0))
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()