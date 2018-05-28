'''
The models package provides some methods for predicting which 
trades to select given some metrics and filtering values.
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


from measures.valuations import ValueRatio

def build_data_table(strat, 
                     epv = ['Adjusted', 'Cyclic', 'Base'], 
                     metrics = ['ROIC (%)', 'Growth mult.', 'Dilution (%)']):

    filter_names = epv[:]
    filter_names.extend(metrics)

    print('Loading filter data', end='')
    filters = []
    for filt in filter_names:
        if filt in epv:
            filters.append(ValueRatio('EPV', filt)(strat))
        elif filt in metrics:
            filters.append(strat.market.get_valuations('Metrics', filt))
        print('.', end = '')

    print('\nBuilding data table', end = '')
    trade_df = strat.trades.as_dataframe(); print('.', end='')
    filter_names = []
    for filter in filters:
        trade_df = strat.trades.add_to_df(trade_df, filter)
        filter_names.append(filter.name)
        print('.', end='')
    print('')

    print('\nCleaning data')
    # Drop any rows with NA values.
    len_before = len(trade_df)
    print('- starting rows: {}'.format(len_before))
    trade_df = trade_df.dropna()
    print('- dropped {} NA rows'.format(len_before - len(trade_df)))
    # Drop rows where filter values are > 2 std devs from mean.
    len_before = len(trade_df)
    for filter_col in filter_names:
        filter_col = trade_df[filter_col]
        filt_mean = filter_col.mean()
        filt_sd = filter_col.std()
        lower_bnd = (filt_mean - 2 * filt_sd)
        upper_bnd = (filt_mean + 2 * filt_sd)
        trade_df = trade_df[(filter_col > lower_bnd) & (filter_col < upper_bnd)]
    print('- dropped {} Outlier rows'.format(len_before - len(trade_df)))
    
    trade_df.filter_names = filter_names
    return trade_df


def prepare_x_y_data(trade_df, filter_names, model_metric, metric_threshold):
    x_table = trade_df[filter_names]
    y_data = trade_df[model_metric].copy()
    y_data.loc[y_data > metric_threshold] = 1
    y_data.loc[y_data <= metric_threshold] = 0
    return (x_table.values, y_data.values)


def build_model_data(strat, 
                     epv = ['Adjusted', 'Cyclic', 'Base'], 
                     metrics = ['ROIC (%)', 'Growth mult.', 'Dilution (%)'], 
                     model_metric = 'normalised_return', 
                     metric_threshold = 0.15):
    
    trade_df = build_data_table(strat, epv, metrics)
    print('Preparing model data')
    x_data, y_data = prepare_x_y_data(trade_df, trade_df.filter_names, model_metric, metric_threshold)
    return (trade_df, x_data, y_data)


def assess_classifier(model, x, y, split_test = 0.3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_test)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    print('Confusion matrix train\n')
    print(confusion_matrix(y_train, train_pred))
    print('\nConfusion matrix test\n')
    print(confusion_matrix(y_test, test_pred))
    print('\nClassification report train\n')
    print(classification_report(y_train, train_pred))
    print('\nClassification report test\n')
    print(classification_report(y_test, test_pred))
    return model

def assess_regressor(model, x, y, split_test = 0.3, threshold = 0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_test)

    if isinstance(model, list):
        first_model = model[0]
        first_model.fit(x_train, y_train)
        train_pred = first_model.predict(x_train)
        test_pred = first_model.predict(x_test)
        for m in model[1:]:
            m.fit(x_train, y_train)
            train_pred += m.predict(x_train)
            test_pred += m.predict(x_test)
        train_pred /= len(model)
        test_pred /= len(model)
    else:
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

    train_pred_id = train_pred[:]
    train_pred_id[train_pred_id <= threshold] = -1
    train_pred_id[train_pred_id > threshold] = 1

    test_pred_id = test_pred[:]
    test_pred_id[test_pred_id <= threshold] = -1
    test_pred_id[test_pred_id > threshold] = 1

    train_id = y_train[:]
    train_id[train_id <= threshold] = -1
    train_id[train_id > threshold] = 1

    test_id = y_test[:]
    test_id[test_id <= threshold] = -1
    test_id[test_id > threshold] = 1

    print('Confusion matrix train\n')
    print(confusion_matrix(train_id, train_pred_id))
    print('\nConfusion matrix test\n')
    print(confusion_matrix(test_id, test_pred_id))
    print('\nClassification report train\n')
    print(classification_report(train_id, train_pred_id))
    print('\nClassification report test\n')
    print(classification_report(test_id, test_pred_id))
    return model
