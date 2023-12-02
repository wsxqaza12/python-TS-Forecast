from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """

    # df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    # df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    # df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear

    # X = df[['hour', 'dayofweek', 'month',
    #         'dayofyear', 'dayofmonth']]
    X = df
    if label:
        y = df[label]
        return X, y
    return X


def get_categorical_indicies(df):
    """
    Indicate category columns
    """

    cats = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            pass
        else:
            cats.append(col)
    cat_indicies = []
    for col in cats:
        cat_indicies.append(df.columns.get_loc(col))
    return cat_indicies


def split_backtest(df, time=3, day=9, date_col='date'):
    """
    Get the time of each interval of Backtesting
    """

    time_end = df[date_col].unique().max()
    time_end = pd.to_datetime(time_end, format='%Y-%m-%d')
    test_start = []
    test_end = []

    for i in range(time):
        split_date = time_end - datetime.timedelta(days=day)
        test_start.append(split_date)
        test_end.append(time_end)
        time_end = split_date

    test_time_range = [test_start, test_end]
    return test_time_range


def split_X_y(df, date_col, y_col):
    """
    Divide the dataframe into two: X and Y
    """
    df = create_features(df)
    x_col = list(set(df) - set(y_col) - set([date_col]))

    y = df[y_col]
    X = df[x_col]

    return X, y


def split_X_y_by_time(df, test_start, test_end,
                      date_col='date', y_col=['d90_arpu']):
    """
    Divide the dataframe into train and test for X and Y respectively
    """
    date_ar = df[date_col]
    X, y = split_X_y(df, date_col, y_col)

    train_row = date_ar <= test_start
    test_row = (date_ar > test_start) & (date_ar <= test_end)
    X_train, X_test = X[train_row], X[test_row]
    y_train, y_test = y[train_row], y[test_row]

    return X_train, X_test, y_train, y_test


def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())],
                        columns=dictionary.keys())


def plot_feature_importance(importance, names, model_type):
    """
    plot feature importance
    """

    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    fig = plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

    return fig
