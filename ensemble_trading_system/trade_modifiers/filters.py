
class FilterInterface:

    def __init__(self, values):
        self.values = values

    def __call__(self, trade):
        if self.accepted_trade(trade):
            return trade
        else:
            return None

    def get(self, date, ticker):
        try:
            value = self.values.loc[date, ticker]
        except KeyError:
            value = None
        return value

    def accepted_trade(self, trade):
        '''
        Each filter must implement a method accepted_trade which accepts a trade object
        and returns a boolean to determine if the trade should be kept.
        '''
        raise NotImplementedError("Filter must implement 'accepted_trade' method")

    def plot(self, ticker, start, end, ax):
        self.values.plot(ticker, start, end, ax)


class HighPassFilter(FilterInterface):
    '''
    HighPassFilter allows trades where the filter values are above the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Filters assume that the supplied values are a DataElement object.
        threshold specifies the value above which the filter passes.
        '''
        super().__init__(values)
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.get(trade.entry, trade.ticker)
        if value is None:
            return False
        else:
            return value > self.threshold


class LowPassFilter(FilterInterface):
    '''
    LowPassFilter allows trades where the filter values are below the specified threshold.
    '''
    def __init__(self, values, threshold):
        '''
        Filters assume that the supplied values are a DataElement object.
        threshold specifies the value below which the filter passes.
        '''
        super().__init__(values)
        self.threshold = threshold
        self.filter_name = values.name
        self.name = '{}:{}'.format(values.name, threshold)

    def accepted_trade(self, trade):
        '''
        Returns True if the value is above the threshold at the trade entry, else False.
        '''
        value = self.get(trade.entry, trade.ticker)
        if value is None:
            return False
        else:
            return value < self.threshold


class BandPassFilter(FilterInterface):
    '''
    BandPassFilter allows trades according to where the filter values lie with respect to the
    specified range. if 'within_bounds' is True (default), values falling within the range result in
    acceptance of the trade, and vice-versa.
    '''
    def __init__(self, values, bounds, within_bounds = True):
        '''
        Assumes that values is a DataElement object.'
        bounds is an iterable (e.g. tuple) which specifies the upper and lower range.
        When within_bounds is True (the default), the filter will pass values between
        the min and max values of bounds, else it will pass when outside the min or max.
        '''
        super().__init__(values)
        self.bounds = bounds
        self.within_bounds = within_bounds
        self.filter_name = values.name
        self.name = '{}:{}-{}'.format(values.name, min(bounds), max(bounds))

    def accepted_trade(self, trade):
        '''
        Returns True if the value is within the threshold at the trade entry, else False.
        '''
        value = self.get(trade.entry, trade.ticker)
        if value is None:
            return False
        else:
            return (not self.within_bounds) ^ (min(self.bounds) < value < max(self.bounds))

