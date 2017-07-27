'''
Created on 22 Dec 2014

@author: Mark
'''
from pandas import DataFrame, Panel
from numpy.matlib import randn
from numpy.random import choice
from pandas.tseries.index import date_range

# Constructors
def buildNumericPanel(items, columns, length):
    panel_dict = {item:buildNumericDataFrame(columns, length) for item in items}
    return Panel(panel_dict)

def buildTextPanel(items, columns, length, factors = ["False", "True"]):
    panel_dict = {item:buildTextDataFrame(columns, length) for item in items}
    return Panel(panel_dict)

def buildNumericDataFrame(columns, length):
    index = date_range("1/1/2010", periods = length, freq = "D")
    return DataFrame(randn(length, len(columns)), index = index, columns = columns)
    
def buildTextDataFrame(columns, length, factors = ["False", "True"]):
    index = date_range("1/1/2010", periods = length, freq = "D")
    return DataFrame(choice(factors, (length, len(columns))), index = index, columns = columns)
        


import time

class Timer(object):
    '''
    Usage:
    with Timer() as t:
        [run method to be timed]
    print("some words: %s s" % t.secs)
    
    or the following will output as milliseconds "elapsed time: [time] ms":
    with Timer(verbose = True) as t:
        [run method to be timed]
    '''
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print("elapsed time: %f ms" % self.msecs)



    
