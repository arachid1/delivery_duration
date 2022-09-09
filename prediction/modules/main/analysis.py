import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

def plot_outliers(df, columns):
    '''
    plot box plots of given columns in a given df
    :param df: dataframe (pandas.DataFrame)
    :param columns: columns to analyze (list of strings)
    '''
    for c in columns:
        if c in df.columns:
            print(c)
            print(df[c].nlargest(n=10))
            boxplt = df.boxplot(column=[c])
            plt.show()


def plot_numerical_columns(df, columns):
    '''
    plot histograms in a multiple subplots at once
    :param df: dataframe (pandas.DataFrame)
    :param columns: columns to analyze (list of strings)
    '''
    fig, axes = plt.subplots(2, 3, figsize=(12, 18))
    fig.suptitle('Distribution of numerical features', fontsize=12)
    for i, c in enumerate(columns):
        if c in df.columns:
            axes[i//3][(i % 3)-1].hist(df[c], bins=50)
            axes[i//3][(i % 3)-1].set_title(c)
            axes[i//3][(i % 3)-1].set_xlabel('values')
            axes[i//3][(i % 3)-1].set_ylabel('counts')


def plot_categorical_columns(df, columns):
    '''
    plots bar plots of distribution of categorical variables
    :param df: dataframe (pandas.DataFrame)
    :param columns: columns to analyze (list of strings)
    '''
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, len(columns)*5))
    fig.suptitle('Distribution of categorical features', fontsize=12)
    for i, c in enumerate(columns):
        df_cat = df[c].value_counts()
        print(df_cat)
        df_cat = df_cat.to_frame().reset_index()
        df_cat.columns = [c, "count"]
        plt.xticks(rotation=35)
        ax = sns.barplot(ax=axes[i], x=df_cat[c], y=df_cat["count"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def apply_rolling_data(data, col, function, window, step=1):
    """Perform a rolling window analysis at the column `col` from `data`

    Given a dataframe `data` with time series, call `function` at
    sections of length `window` at the data of column `col`. Append
    the results to `data` at a new columns with name `label`.

    Parameters
    ----------
    data : DataFrame
        Data to be analyzed, the dataframe must stores time series
        columnwise, i.e., each column represent a time series and each
        row a time index
    col : str
        Name of the column from `data` to be analyzed
    function : callable
        Function to be called to calculate the rolling window
        analysis, the function must receive as input an array or
        pandas series. Its output must be either a number or a pandas
        series
    window : int
        length of the window to perform the analysis
    step : int
        step to take between two consecutive windows

    Returns
    -------
    data : DataFrame
        Input dataframe with added columns with the result of the
        analysis performed

    """

    x = _strided_app(data[col].to_numpy(), window, step)
    rolled = np.apply_along_axis(function, 1, x)
    return rolled


def _strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """returns an array that is strided
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S*n, n))


def run_linear_regression(df, y_hist, dropped_features):
    '''
    function to execute a linear regression analysis on dataframe, with the option to drop columns
    :param df: dataframe (pandas.DataFrame)
    :param y_hist: dataset labels (pandas.DataFrame)
    :param dropped_features: columns to drop (list of strings)
    '''
    temp = df
    temp = temp.drop(columns=dropped_features)
    X_hist = temp.to_numpy()
    X_hist = X_hist
    regr = linear_model.LinearRegression()
    regr = regr.fit(X_hist, y_hist)
    y_pred = regr.predict(X_hist)
    mae = mean_absolute_error(y_hist, y_pred)
    print("mae: {}".format(mae))
    print("coefficients")
    o_is = regr.coef_.argsort()
    for ordered_i in o_is:
        print("{} : {}".format(temp.columns[ordered_i],regr.coef_[ordered_i]))