
# TODO we shouldn't need a specific type for filter data. However a type which handles stacked values would be useful in general.

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
        df = self.values.copy()
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

    def value_ratio(self, market):
        fullDF = market.get_empty_dataframe()
        values = self.values.reindex(fullDF.index, method = 'ffill')
        ratios = values / market.close
        return ratios

    def value_rank(self, market):
        ratios = self.value_ratio(market)
        ranks = ratios.rank(axis = 1, ascending = False)
        return ranks


class ValueFilterValues(StackedFilterValues):
    '''
    Returns the ratio of the calculated value to the price at time of entry.
    '''
    def get_for_df(self, df, row):
        ticker = df.ticker[row]
        date = df.entry[row]
        recent = self.get(ticker, date)
        price = df.entry_price[row]
        if recent is not None:
            return recent / price
        else:
            return recent
