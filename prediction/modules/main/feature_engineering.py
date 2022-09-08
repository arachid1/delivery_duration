from . import parameters as p
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import time
from collections import defaultdict
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn'


def add_features(df, label="historical_", search_spaces=None):
    """
    extends given dataset with new features determined by functions and their configs, such as p.domain_settings
    finds new ways to represent data, such as time, or correlations in the vast dataset along certain characteristics,
    such as a strong predictor or time-dependent features
    is capable of searching and applying rules flexibly to other search spaces
    :param df: dataset (pandas.dataFrame)
    :param label: label to distinguish historical and test features (string)
    :param search_spaces: dictionary that details which datasets to use and how to include them (dict)
    :returns: the dataset with all new features/columns (pandas.dataFrame)
    """
    df = add_time_features(df)
    df = add_domain_features(df, label, search_spaces)

    return df


def add_holiday(df, c):
    '''
    extends dataframe with a column which indicate if the order (row) was created on a holiday
    :param df: dataframe (pandas.DataFrame)
    :param c: column name indicating the date (string)
    :return: df with additional holiday column
    '''
    cal = USFederalHolidayCalendar()
    start = df['created_at'].min()
    end = df['created_at'].max()
    holidays = cal.holidays(start=start, end=end).to_pydatetime()
    df['holiday'] = df[c].isin(holidays).astype(int)
    return df


def add_time_features(df):
    """
    generates new features from the time the order was created, such as day of the week
    :param df: dataset (pandas.dataFrame)
    :returns: the dataset with new columns (pandas.dataFrame)
    """
    c = "created_at"

    df["week_day"] = df[c].dt.dayofweek
    df['weekend'] = (df[c].dt.dayofweek > 4).astype(int)
    df["day_of_month"] = df[c].dt.day
    df["month"] = df[c].dt.month
    df["hour"] = df[c].dt.hour

    b = [0, 4, 8, 12, 16, 20, 24]
    l = [0, 1, 2, 3, 4, 5]  # Late Night, Early Morning, Morning, Noon, Eve, Night
    df["day_part"] = pd.cut(df['hour'], bins=b, labels=l, include_lowest=True).astype('int64')
    # df = add_holiday(df, c) # unfortunately, no holidays during the dataset timeframe

    return df


def add_domain_features(df, label, search_spaces):
    """
    handles new domain features with IO system for efficiency/correctness.
    generates all features from one domain first, then moves to the next
    => therefore, writes/reads each domain features to a separate domain file (write mode with p.feature_writing = True)
    ensures the new columns are written and stored in the same order

    :param df: dataset (pandas.dataFrame)
    :param label: label indicating historical or test dataset (string)
    :param search_spaces: dictionary that details which datasets to use and how to include them (dict)
    :returns: the dataset with new columns (pandas.dataFrame)
    """
    for i, domain in enumerate(p.domain_columns):
        # this sorting ensures read/writing are in the same order
        df.sort_values(by=[domain, 'created_at'], ascending=[True, True],
                       inplace=True)
        df = df.reset_index(drop=True)
        file_name = str(domain + "_features")
        if p.debugging:
            file_name = str("testing_" + file_name)
        features_path = os.path.join(p.data_root, str(
            label + file_name + p.feature_version + ".csv"))  # creating file name
        if p.feature_writing or p.debugging:
            _all = gen(df, domain, search_spaces)
            write_domain_feature(_all, features_path)
        df = read_domain_feature(df, features_path)
    return df


def gen(df, domain, search_spaces):
    """

    Given domain to search, generates all features for for all samples by finding trends, scores, etc determined by p.domain_settings
    example: average of the subtotal of the orders in the last 3 hours
    reduces search space to corresponding domain only for less complexity
    :param df: dataset (pandas.dataFrame)
    :param domain: name of domain to generate features with (string)
    :param search_spaces: dictionary that details which datasets to use and how to include them (dict)
    :returns: dictionary with feature name as keys and lists as values, with lists following the original order from df (dict)
    """

    start_time = time.time()
    _all = defaultdict(lambda: [])  # features dictionary
    for v in sorted(df[domain].unique()):
        # reduces search space to same domain value
        domain_df = df.loc[df[domain] == v]
        domain_df.sort_values(by=['created_at'], ascending=[True], inplace=True)
        for i, (feature_name, func, n_hours, space_key, space_names) in enumerate(p.domain_settings):
            search_df = domain_df  # default search space
            if search_spaces is not None and space_key is not 'default':
                if space_key is 'add':  # combined search sapace
                    # TODO: add unittests
                    for space_name in space_names:
                        space = search_spaces[space_name]
                        space = space[domain_df.columns.intersection(space.columns)]
                        search_df = pd.concat([search_df, space], ignore_index=True, axis=0)
                else:  # selected differing search space
                    search_df = search_spaces[space_key]

            search_df = search_df.loc[search_df[domain] == v]

            feature_name = str(feature_name + domain)
            domain_features = domain_df.apply(
                lambda row: func(search_df, row, n_hours),
                axis=1).tolist()
            _all[feature_name].extend(domain_features)
    print('--- {} features generated in {} secs. ---'.format(domain, time.time()-start_time))
    return _all


def write_domain_feature(_all, features_path):
    """
    writes domain-dependent feature to a csv file (to be later read and appended as a new column to the dataset)
    order of _all is handled in parent loop
    :param _all: dictionary of all features in the order of the data frame / in which they were written (dict)
    :param features_path: name of domain path to write features from (string)

    """
    print("--- Writing {}...".format(features_path))
    domain_features = pd.DataFrame(
        _all)
    domain_features.to_csv(features_path, index=False)
    if not p.debugging:
        domain_features.to_csv(
            features_path + ".{}".format(datetime.now().strftime("%m_%d_%Y")),
            index=False)  # writing an extra copy in case


def read_domain_feature(df, features_path):
    """
    reads domain-dependent features from a csv file and inserts them as a column to a data frame
    order of df and features read is handled in parent loop
    :param df: dataset (pandas.dataFrame)
    :param features_path: path to the csv file to read features from (string)
    :return: the new dataframe with the  additional feature/columns read (pandas.DataFrame)
    """
    print("--- Reading {}...".format(features_path))
    n_rows = df.shape[0]
    features_data = pd.read_csv(features_path)
    if p.debugging:
        features_data = features_data[:df.shape[0]]
        features_path = str(features_path + "_debugging")
    n_domain_rows = features_data.shape[0]
    assert n_rows == n_domain_rows, "mismatch between n_rows of data frame: {} and n_rows in new feature column: {}".format(
        n_rows, n_domain_rows)
    df = df.join(features_data)
    return df


def return_prev_orders(df, row, n_hours, completed_only=False):
    """
    returns the orders that happened in the last n_hours before the current order
    :param df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :param completed_only: indicates whether or not to return previous orders that finished only (bool)
    :return: orders that occured and/or completed in the last n_hours (pandas.dataFrame)
    """
    # orders that happened before the current
    prev_orders = df.loc[(row['created_at'] - df['created_at']).dt.total_seconds() > 0]

    # orders that happened less than n_hours ago
    prev_orders = prev_orders.loc[(row['created_at'] - df['created_at']
                                   ).dt.total_seconds() <= 3600 * n_hours]

    if completed_only:
        prev_orders = prev_orders.loc[prev_orders['actual_delivery_time'] <=
                                      row['created_at']]  # completed orders only

    return prev_orders


def return_count(domain_df, row, n_hours):
    """
    returns the number of orders prior to the current order
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: the count of previous orders (int)
    """
    return len(return_prev_orders(domain_df, row, n_hours))


def return_avg_subtotal(domain_df, row, n_hours):
    """
    returns the average subtotal of the orders prior to the current order
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: the average subtotal of the previous orders (float)
    """
    sub_df = return_prev_orders(domain_df, row, n_hours)
    if sub_df.empty:
        return 0.0
    else:
        return float(sub_df['subtotal'].mean())


def return_avg_time(domain_df, row, n_hours):
    """
    returns the average delivery duration of the orders that completed prior to the start of the current order
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: the average duration of the previous orders that completed (float)
    """
    sub_df = return_prev_orders(domain_df, row, n_hours, completed_only=True)

    if len(sub_df) < 2:  # minimum 2 to avoid rewriting label of previous samples and bias model
        return 0.0
    else:
        return float(sub_df['delivery_duration_sec'].mean())


def return_long_score(domain_df, row, n_hours=730):  # 24*30
    """
    scores historical rate at which a domain (i.e., a market) has long orders using p.long_threshold deriver from quantiles
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: historical rate of long orders (float)
    """
    sub_df = return_prev_orders(domain_df, row, n_hours, completed_only=True)
    if len(sub_df) == 0:
        return 0.0
    long = sub_df.loc[sub_df['delivery_duration_sec'] >= p.long_threshold]
    if len(long) == 0:
        return 0.0
    return round(len(long)/len(sub_df), 2)


def return_short_score(domain_df, row, n_hours=730):
    """
    scores historical rate at which a domain (i.e., a market) has short orders using p.short_threshold deriver from quantiles
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: historical rate of short orders (float)
    """
    sub_df = return_prev_orders(domain_df, row, n_hours, completed_only=True)
    if len(sub_df) == 0:
        return 0.0
    short = sub_df.loc[sub_df['delivery_duration_sec'] <= p.short_threshold]
    if len(short) == 0:
        return 0.0
    return round(len(short)/len(sub_df), 2)


def return_trip_to_customer(domain_df, row, n_hours=730):
    """
    uses a linear regression model to predict total trip to customer from X=estimated_store_to_consumer_driving_duration
    trains on previous orders
    :param domain_df: dataset (pandas.dataFrame)
    :param row: current order (pandas.dataSeries)
    :param n_hours: number of hours prior to the current order (int)
    :return: predicted delivery_duration (float)
    """
    sub_df = return_prev_orders(domain_df, row, n_hours)
    if len(sub_df) < 2:  # same precaution as seen in multiple functions above
        return 0
    X = sub_df.loc[:, ['estimated_store_to_consumer_driving_duration']].to_numpy().reshape(-1, 1)
    y = sub_df.loc[:, ['delivery_duration_sec']].to_numpy().reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr = regr.fit(X, y)
    curr = np.array(row.loc['estimated_store_to_consumer_driving_duration'])
    curr = np.reshape(curr, newshape=(-1, 1))
    return int(regr.predict(curr))
