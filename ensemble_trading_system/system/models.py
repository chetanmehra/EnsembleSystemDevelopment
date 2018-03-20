'''
The models package provides some methods for predicting which 
trades to select given some metrics and filtering values.
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


from measures.valuations import ValueRatio

def build_model_data(strat, 
                     epv = ['Adjusted', 'Cyclic', 'Base'], 
                     metrics = ['ROIC (%)', 'Growth mult.', 'Dilution (%)'], 
                     model_metric = 'normalised_return', 
                     metric_threshold = 0.15):
    
    filter_names = epv[:]
    filter_names.extend(metrics)

    print('Loading filter data', end='')
    filters = {}
    for filt in filter_names:
        if filt in epv:
            filters[filt] = ValueRatio('EPV', filt)(strat)
        elif filt in metrics:
            filters[filt] = strat.market.get_valuations('Metrics', filt)
        print('.', end = '')

    print('\nBuilding data table', end = '')
    trade_df = strat.trades.as_dataframe(); print('.', end='')
    processed_filter_names = []
    for key, val in filters.items():
        trade_df = strat.trades.add_to_df(trade_df, val)
        processed_filter_names.append(val.name)
        print('.', end='')
    print('')

    print('\nCleaning data')
    # Drop any rows with NA values.
    trade_df = trade_df.dropna()
    # Drop rows with annualised returns which might skew the results.
    trade_df = trade_df[trade_df.annualised_return < 1000]

    print('Preparing model data')
    x_table = trade_df[processed_filter_names]
    y_data = trade_df[model_metric]
    y_data[y_data > metric_threshold] = 1
    y_data[y_data <= metric_threshold] = 0

    return (x_table.values, y_data.values)


def example(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_train_pred = lr.predict(x_train)
    lr_test_pred = lr.predict(x_test)

    print('Confusion matrix train\n')
    print(confusion_matrix(y_train, lr_train_pred))
    print('\nConfusion matrix test\n')
    print(confusion_matrix(y_test, lr_test_pred))
    print('\nClassification report train\n')
    print(classification_report(y_train, lr_train_pred))
    print('\nClassification report test\n')
    print(classification_report(y_test, lr_test_pred))

    

