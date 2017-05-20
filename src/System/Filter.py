

class FilterInterface():

    def __call__(self, strategy):
        raise NotImplementedError("Filter must be callable")

    def plot(self, ticker, start, end, ax):
        raise NotImplementedError("Filter must implement 'plot' method")


class FilterValues:

    def __init__(self, values, name = None):
        '''
        values should be a dataframe with index of dates
        '''
        self.values = values
        if name is None:
            self.name = "Filter"
        else:
            self.name = name


    def __getitem__(self, key):
        raise NotImplementedError

    def get(self, df, row):
        relevant_values = self.find_relevant_values(df, row)
        return self.most_recent(relevant_values)

    def find_relevant_values(self, df, row):
        ticker = df.ticker[row]
        date = df.entry[row]
        tick_values = self[ticker]
        return tick_values[tick_values.index < date]


    def most_recent(self, relevant_values):
        if relevant_values.empty:
            result = None
        elif relevant_values.shape[1] == 1:
            result = relevant_values.iloc[-1].values[0]
        else:
            result = relevant_values.iloc[-1].values
        return result

    def plot(self, ticker, start = None, end = None, ax = None):
        df = self.get_plot_frame()
        df[ticker][start:end].plot(ax = ax)

    def get_plot_frame(self):
        return self.values


class StackedFilterValues(FilterValues):
    '''
    A stacked filter may contain more than one filter type, with a column for
    ticker, and each of the filter types.
    Each row denotes the related ticker.
    '''

    def __getitem__(self, key):
        return self.values[self.values.ticker == key][self.values.columns[1:]]

    @property
    def types(self):
        return self.values.columns[1:].tolist()

    def as_wide_values(self, type = None):
        if type is None:
            type = self.types[0]
        df = self.values
        df['date'] = df.index
        df = df.pivot(index = 'date', columns = 'ticker', values = type)
        df = df.fillna(method = 'ffill')
        return WideFilterValues(df, name = type)

    def get_plot_frame(self):
        return self.as_wide_values().values

class WideFilterValues(FilterValues):

    '''
    Wide filters contain one type of filter with a column for each ticker.
    '''

    def __getitem__(self, key):
        return self.values[[key]]

    @property
    def types(self):
        return [self.name]


class ValueFilterValues(StackedFilterValues):
    '''
    Returns the ratio of the calculated value to the price at time of entry.
    '''

    def get(self, df, row):
        relevant_values = self.find_relevant_values(df, row)
        price = df.entry_price[row]
        recent = self.most_recent(relevant_values)
        if recent is not None:
            return recent / price
        else:
            return recent
