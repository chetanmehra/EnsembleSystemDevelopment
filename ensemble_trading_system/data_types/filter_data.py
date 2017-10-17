
# TODO we shouldn't need a specific type for filter data. However a type which handles stacked values would be useful in general.

class FilterValues:

    def __init__(self, values, name = None):
        '''
        values should be a dataframe with index of dates
        '''
        self.data = values
        if name is None:
            self.name = "Filter"
        else:
            self.name = name

    def __getitem__(self, key):
        raise NotImplementedError

    def get(self, ticker, date):
        relevant_values = self.find_relevant_values(ticker, date)
        return self.most_recent(relevant_values)

    def get_for_df(self, df, row):
        ticker = df.ticker[row]
        date = df.entry[row]
        return self.get(ticker, date)

    def find_relevant_values(self, ticker, date):
        tick_values = self[ticker]
        return tick_values[tick_values.index < date]

    def most_recent(self, relevant_values):
        if relevant_values.empty:
            result = None
        elif relevant_values.shape[1] == 1:
            result = relevant_values.iloc[-1].data[0]
        else:
            result = relevant_values.iloc[-1].data
        return result

    def plot(self, ticker, start = None, end = None, ax = None):
        df = self.get_plot_frame()
        df[ticker][start:end].plot(ax = ax)

    def get_plot_frame(self):
        return self.data


class StackedFilterValues(FilterValues):
    '''
    A stacked filter may contain more than one filter type, with a column for
    ticker, and each of the filter types.
    '''

    def __getitem__(self, key):
        return self.data[self.data.ticker == key][self.data.columns[1:]]

    @property
    def types(self):
        return self.data.columns[1:].tolist()

    def as_wide_values(self, type = None, index = None):
        if type is None:
            type = self.types[0]
        df = self.data.copy()
        df['date'] = df.index
        df = df.pivot(index = 'date', columns = 'ticker', values = type)
        df = df.fillna(method = 'ffill')
        if index is not None:
            df = df.reindex(index, method = 'ffill')
        return WideFilterValues(df, name = type)

    def get_plot_frame(self):
        return self.as_wide_values().data


class WideFilterValues(FilterValues):

    '''
    Wide filters contain one type of filter with a column for each ticker.
    '''
    def __getitem__(self, key):
        return self.data[[key]]

    @property
    def types(self):
        return [self.name]

    def value_ratio(self, prices):
        values = self.data.reindex(prices.index, method = 'ffill')
        ratios = values / prices - 1
        return WideFilterValues(ratios, self.name + "_ratio")

    def value_rank(self, prices):
        ratios = self.value_ratio(prices)
        ranks = ratios.rank(axis = 1, ascending = False)
        return WideFilterValues(ranks, self.name + "_rank")

