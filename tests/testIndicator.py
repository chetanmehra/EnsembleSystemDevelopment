'''
Created on 13 Dec 2014

@author: Mark
'''
import unittest
from System.Indicator import Signal, Crossover
from pandas import DataFrame
from numpy import NaN, isnan
from tests.TestHelpers import buildTextDataFrame, buildNumericDataFrame
from mock import Mock
from pandas.stats.moments import ewma


class TestIndicatorInterface(unittest.TestCase):
    
    def setUp(self):
        tickers = ["ASX", "BHP"]
        self.factors = ["False", "True"]
        self.data = buildTextDataFrame(tickers, 10, factors = self.factors)
        self.indicator = Signal(self.data)
        
    def testIndicatorReturnsLevels(self):
        self.assertEqual(self.indicator.levels, self.factors)
        
    def testIndicatorReturnsLevelsWhenMissingValues(self):
        self.data.iloc[0] = NaN
        indicator = Signal(self.data)
        self.assertEqual(indicator.levels, self.factors)
        
    def testIndicatorCachesLevels(self):
        self.assertIs(self.indicator._levels, None)
        self.indicator.levels
        self.assertEqual(self.indicator._levels, self.factors)
        
    def testIndicatorIndexByTicker(self):
        self.assertIs(self.indicator["ASX"], self.indicator.data["ASX"])
        
    def testIndicatorLength(self):
        self.assertEqual(len(self.indicator), len(self.data))
        
    def testIndicatorIndex(self):
        self.assertTrue(all(self.indicator.index == self.data.index))


class TestMeasure(unittest.TestCase):
    
    def setUp(self):
        self.prices = buildNumericDataFrame(["ASX", "BHP", "CBA"], 10).abs()
        self.strategy = Mock()
        self.strategy.get_indicator_prices = Mock(return_value = self.prices)
        self.crossover = Crossover(5, 3)
        self.indicator = self.crossover(self.strategy)
        
    def testEMAhandlesMissingValues(self):
        window = 4
        self.prices["ASX"][2] = NaN
        self.prices["BHP"][0] = NaN
        self.prices["CBA"][3:5] = NaN
        ema = ewma(self.prices, span = window)
        self.assertFalse(isnan(ema["ASX"][2]))
        self.assertFalse(isnan(ema["ASX"][3]))
        self.assertTrue(isnan(ema["BHP"][0]))
        self.assertFalse(isnan(ema["BHP"][1]))
        self.assertFalse(any(isnan(ema["CBA"][3:5])))
        
    def testCrossoverReturnsIndicator(self):
        self.assertIsInstance(self.indicator, Signal)
        
    def testCrossoverDataIsStringType(self):
        self.assertIsInstance(self.indicator.data, DataFrame)
        self.assertTrue(all(self.indicator.data.dtypes == "object"), self.indicator.data.dtypes)
        
    def testCrossoverReturnsSameShapeAsMarket(self):
        self.assertEqual(self.indicator.data.shape, self.prices.shape)
        
    def testCrossoverUpdatesParameters(self):
        self.assertEqual(self.crossover.slow, 5)
        self.assertEqual(self.crossover.fast, 3)
        self.crossover.update_param((10, 5))
        self.assertEqual(self.crossover.slow, 10)
        self.assertEqual(self.crossover.fast, 5)
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()