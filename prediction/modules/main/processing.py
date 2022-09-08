import pandas as pd
from . import parameters as p


def extract_samples(file_path, columns):
    """
    this functions reads and preprocesses data, readying it for training or prediction
    it also infers label values for the historical data
    :param file_path: path to data source file (string)
    :param columns: columns representing time variables (list of strings)
    :return: dataset with rows as samples and features as columns (Pandas.dataFrame)
    """
    df = pd.read_csv(file_path)
    df = clean_data(df, ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders'])
    df = process_timestamps(df, columns)

    if 'actual_delivery_time' in df.columns:  # checking for train, not test dataset
        df['delivery_duration_sec'] = (
            df['actual_delivery_time'] - df['created_at']).dt.total_seconds()  # label
        df = remove_outliers(df)
    if p.debugging:  # returning smaller dataset in testing mode
        df = df[:100]
    return df


def process_timestamps(df, columns):
    """
    this functions converts columns representing time to pandas datetime object
    :param df: dataset (Pandas.dataFrame)
    :param columns: name of columns to convert (list of strings)
    :return: data with converted column values (Pandas.dataFrame)
    """
    for c in columns:
        df[c] = pd.to_datetime(df[c])
    return df


def clean_data(df, columns):
    """
    this functions removes rows that contains negative or NA values
    :param df: dataset (Pandas.dataFrame)
    :param columns: columns that contain negative and/or NA values (list of strings)
    :return: data with rows considered unusable removed (Pandas.dataFrame)
    """
    # possible TODO: check for inconsistent on_shift/busy dashers numbers + inconsistent delivery_duration_sec/estimated_duration
    for c in columns:
        df = df.drop(df.index[df[c] < 0])
    df.dropna(inplace=True)
    assert df.isnull().sum().sum() == 0, "dataset still contains null values"
    return df


def remove_outliers(df, n_hours=3):
    """
    this functions removes rows with extremely long delivery times
    they can be considered poor examples for training
    :param df: dataset (Pandas.dataFrame)
    :param n_hours: threshold for delivery length in hours (int)
    :return: data with rows considered outliers removed (Pandas.dataFrame)
    """
    df = df.drop(df.index[df['delivery_duration_sec'] >= 3600*n_hours]
                 )  # 3600 is number of seconds in an hour
    return df
