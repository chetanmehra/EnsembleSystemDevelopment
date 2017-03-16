'''
Created on 6 Dec 2014

@author: Mark
'''

import pandas_datareader
import pickle

class Handler(object):
    '''
    Handler uses the Pandas data functionality to download data and handle local storage.
    '''


    def __init__(self, location):
        '''
        Constructor
        '''
        self.location = location
        
    def get(self, ticker, start, end):
        self.ticker = ticker
        return pandas_datareader.get_data_yahoo(ticker, start, end)
    
    
    def save(self, item, filename):
        file = open(self.location + filename + ".py", "wb")
        try:
            pickle.dump(item, file)
        finally:
            file.close()
        
    def load(self, filename):
        return pickle.load(open(self.location + filename + ".py", "rb"))
    