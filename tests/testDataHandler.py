'''
Created on 6 Dec 2014

@author: Mark
'''
import unittest
import System.Data as Data
import datetime
import pandas.io.data
from pandas.core.frame import DataFrame

class CheckDownloaderSettings(unittest.TestCase):
    
    def setUp(self):
        self.storage_location = "D:/Data/MarketData/Stocks/Python"
        self.downloader = Data.Handler(self.storage_location)
        self.ticker = "ASX.AX"
    
    def testStorageLocation(self):
        self.assertEquals(self.downloader.location, self.storage_location)
        
    def testDataLoaded(self):
        start = datetime.datetime(2006, 10, 1)
        end = datetime.datetime(2010, 1, 1)
        downloaded_df = self.downloader.get(self.ticker, start, end)
        expected_df = pandas.io.data.get_data_yahoo(self.ticker, start, end)
        self.assertIsInstance(downloaded_df, DataFrame)
        self.assertEquals(len(expected_df), len(downloaded_df))
        self.assertEquals(self.downloader.ticker, self.ticker)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()