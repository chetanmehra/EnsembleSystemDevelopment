
# Analysis of Filter performance with trades

class FilterPerformance():

    def __init__(self, trades):
        self.trade_df = trades.as_dataframe()
        self.result = None

    def add_filter_to_df(self, *args):
        for filter in args:
            cols = filter.types
            for col in cols:
                self.trade_df[col] = None
            if len(cols) == 1:
                cols = cols[0]
            for i in df.index:
                self.trade_df.loc[i, cols] = filter.get(df, i)

    def filter_summary(self, filter_values, bins = 5):
        '''
        filter_values is a FilterValue object. 
        Note if filter_values contains more than one type, only the first is used.
        bins is an iterable of boundary points e.g. (-1, 0, 0.5, 1, etc...), or an integer of 
        the number of bins to produce (default 5). This is passed to pandas qcut.
        '''
        self.add_filter_to_df(filter_values)
        if isinstance(bins, int):
            type_bins = qcut(self.trade_df[filter_values.types[0]], bins)
        else:
            type_bins = cut(self.trade_df[filter_values.types[0]], bins)
        mu = self.trade_df.groupby(type_bins).base_return.mean()
        sd = self.trade_df.groupby(type_bins).base_return.std()
        N = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}
        
    def filter_grouping(self, filter, bins):
        '''
        Provides a summary of filter performance for provided bins. Bins must be a sequence of boundary
        points e.g. (-1, 0, 0.25...). Each filter type will be provided as a column.
        '''
        if isinstance(bins, int):
            raise ValueError("Bins must be a sequence for filter grouping")
        self.add_filter_to_df(filter)
        mu = DataFrame()
        sd = DataFrame()
        N = DataFrame()
        for type in filter.types:
            type_bins = cut(self.trade_df[type], bins)
            mu[type] = self.trade_df.groupby(type_bins).base_return.mean()
            sd[type] = self.trade_df.groupby(type_bins).base_return.std()
            N[type] = self.trade_df.groupby(type_bins).base_return.count()
        self.result = {"mean" : mu, "std" : sd, "count" : N}


    def filter_comparison(self, filter1, filter2, bins1 = 5, bins2 = 5):
        '''
        Provides a matrix comparing mean, std dev, and count for each combination of filter
        values. Note only the first type of each filter is considered.
        '''
        self.add_filter_to_df(filter1, filter2)

        f1_name = filter1.types[0]
        f2_name = filter2.types[0]
        
        if isinstance(bins1, int):
            f1_bins = qcut(self.trade_df[f1_name], bins1)
        else:
            f1_bins = cut(self.trade_df[f1_name], bins1)

        if isinstance(bins2, int):
            f2_bins = qcut(self.trade_df[f2_name], bins2)
        else:
            f2_bins = cut(self.trade_df[f2_name], bins2)

        grouping = self.trade_df.groupby([f1_bins, f2_bins]).base_return

        mu = DataFrame(grouping.mean()).unstack()
        sd = DataFrame(grouping.std()).unstack()
        N = DataFrame(grouping.count()).unstack()

        self.result = {"mean" : mu, "std" : sd, "count" : N}

    def plot_Sharpe(self):
        Sharpe = self.result['mean'] / self.result['std']
        Sharpe.plot()

    def plot(self):
        mean_plus = self.result['mean'] + self.result['std']
        mean_minus = self.result['mean'] - self.result['std']
        self.result['mean'].plot()
        mean_plus.plot(style = '-')
        mean_minus.plot(style = '-')



# Trade Summary Report

