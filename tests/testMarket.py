'''
Created on 6 Dec 2014

@author: Mark
'''
import unittest
import System.Market as Market
from System.Strategy import Indexer
from datetime import datetime
from mock import Mock, call
from pandas import Series, DataFrame
from numpy.matlib import randn
from pandas.core.common import isnull


class TestMarketStructure(unittest.TestCase):

    def setUp(self):
        self.tickers = ["ASX", "BHP", "CBA"]
        self.start_date = datetime(2006, 10, 1)
        self.end_date = datetime(2010, 1, 1)
        self.market = Market.Market(self.tickers, self.start_date, self.end_date)
        
    def testType(self):
        self.assertIsInstance(self.market, Market.Market)
        
    def testTickersAndDates(self):
        self.assertEqual(self.market.tickers, self.tickers)
        self.assertEqual(self.market.start, self.start_date)
        self.assertEqual(self.market.end, self.end_date)
        
    def testReportsOnEmptyDataStatus(self):
        expected_status = dict(zip(self.tickers, [False] * len(self.tickers)))
        self.assertEqual(self.market.status, expected_status)
        
    def testDownloadsConstituentsUpdatesStatus(self):
        self.market.downloader.get = Mock()
        self.market.downloader.get.return_value = DataFrame(randn(2, 1))
        self.market.download_data()
        expected_status = dict(zip(self.tickers, [True] * len(self.tickers)))
        self.assertEqual(self.market.status, expected_status)
        
    def testDownloadConsituentsGrabsData(self):
        self.market.downloader.get = Mock()
        self.market.downloader.get.return_value = DataFrame(randn(2, 1))
        calls = [call(ticker + ".AX", self.start_date, self.end_date) for ticker in self.tickers] 
        self.market.download_data()
        self.market.downloader.get.assert_has_calls(calls)
        self.assertSetEqual(set(self.tickers), set(self.market.instruments.keys()))
        

class TestMarketQueries(unittest.TestCase):
    
    def setUp(self):
        self.tickers = ["ASX", "BHP"]
        self.start_date = datetime(2006, 10, 1)
        self.end_date = datetime(2010, 1, 1)
        self.market = Market.Market(self.tickers, self.start_date, self.end_date)
        self.market["ASX"] = self.market.downloader.load("ASX")
        self.market["BHP"] = self.market.downloader.load("BHP")
        
    def testGetOHLCVDataForTicker(self):
        asx_open = self.market["ASX"]["Open"]
        self.assertIsInstance(asx_open, Series, asx_open.__class__)
        
    def testGetOHLCVDataForAll(self):
        expected_open = self.market.instruments.minor_xs("Open")
        expected_close = self.market.instruments.minor_xs("Close")
        expected_volume = self.market.instruments.minor_xs("Volume")
        self.assertTrue(self.market.open.equals(expected_open))
        self.assertTrue(self.market.close.equals(expected_close))
        self.assertTrue(self.market.volume.equals(expected_volume))
        
    def testGettingMarketReturns(self):
        def expected_returns(market, trade_open, trade_close):
            if (trade_open == "Open") & (trade_close == "Close"):
                lag = 0
            else:
                lag = 1
            returns = (market.instruments.minor_xs(trade_close) /
                               market.instruments.minor_xs(trade_open).shift(lag)) - 1
            returns[isnull(returns)] = 0
            return returns
        
        timing_map = {"O":"Open", "C":"Close"}
        
        for timing in ["OO", "CC", "CO", "OC"]:
            indexer = Indexer(timing, "C")
            returns = self.market.returns(indexer)
            expected = expected_returns(self.market, timing_map[timing[0]], timing_map[timing[1]])
            for ticker in self.market.tickers:
                message = ticker + " - " + timing
                self.assertIsInstance(returns[ticker], Series, message)
                self.assertTrue(expected[ticker].equals(returns[ticker]), message)
                
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testStarting']
    unittest.main()