"""
financial_time_series_functions.py
"""

import pandas as pd


def holding_period_return(multivariate_df: pd.DataFrame, lag: int, ending_lag: int = 0, skip_nan: bool = False):
    """
    Calculate the rolling holding period return for each column
    Holding period return for stock = Price(t - ending_lag) / Price(t - lag) - 1
    :param multivariate_df: DataFrame
    :param lag: int
    :param ending_lag: int used to shift the final observation backwards
    :param skip_nan: bool
    :return: DataFrame
    """
    return _general_dataframe_function(multivariate_df=multivariate_df, func=_holding_period_return, return_lag=None,
                                       skip_nan=skip_nan, lag=lag, ending_lag=ending_lag)


def _holding_period_return(multivariate_df: pd.DataFrame, func_param: dict):
    lag_p = 'lag'
    ending_lag_p = 'ending_lag'
    check_required_keys([lag_p, ending_lag_p], func_param)
    if func_param[ending_lag_p] < 0:
        raise ValueError("'ending_lag' needs to be an int larger or equal to zero")
    return multivariate_df.shift(func_param[ending_lag_p]).pct_change(func_param[lag_p] - func_param[ending_lag_p])


def simple_moving_average(multivariate_df: pd.DataFrame, skip_nan: bool = False, window: int = None):
    """
    Calculate a rolling average over a specific window 
    :param multivariate_df: DataFrame
    :param skip_nan: bool
    :param window: int
    :return: DataFrame
    """
    return _general_dataframe_function(multivariate_df=multivariate_df, func=_simple_moving_average, return_lag=None,
                                       skip_nan=skip_nan, window=window)


def _simple_moving_average(multivariate_df: pd.DataFrame, func_param: dict):
    # name the required keys to be used to lookup the parameters
    window_p = 'window'
    check_required_keys([window_p], func_param)
    return get_rolling_object(multivariate_df=multivariate_df, window=func_param[window_p]).mean()


def realized_volatility(multivariate_df: pd.DataFrame, return_lag: int = None, skip_nan: bool = False, window: int = None,
                        annualization_factor: float = 252):
    """
    Calculate the rolling annualized realized volatility (standard deviation of returns)
    :param multivariate_df: DataFrame
    :param return_lag: int
    :param skip_nan: bool
    :param window: int
    :param annualization_factor: float
    :return: DataFrame
    """
    return _general_dataframe_function(multivariate_df=multivariate_df, func=_realized_volatility, return_lag=return_lag,
                                       skip_nan=skip_nan, window=window, annualization_factor=annualization_factor)


def _realized_volatility(multivariate_df: pd.DataFrame, func_param: dict):
    # name the required keys to be used to lookup the parameters
    window_p = 'window'
    annualization_factor_p = 'annualization_factor'
    check_required_keys([window_p, annualization_factor_p], func_param)
    return get_rolling_object(multivariate_df=multivariate_df, window=func_param[window_p]).std() * (func_param[annualization_factor_p] ** 0.5)


def realized_beta(multivariate_df: pd.DataFrame, beta_instrument_name: str, return_lag: int = None, skip_nan: bool = False, window: int = None):
    """
    Calculate the beta for each column with respect to a specific series (a column in the given multivariate DataFrame)
    Beta(i, j) = Covariance(ith instrument, jth instrument) / Variance(jth instrument)
    :param multivariate_df: DataFrame
    :param beta_instrument_name: str
    :param return_lag: int
    :param skip_nan: bool
    :param window: int
    :return: DataFrame
    """
    try:
        beta_instrument_series = multivariate_df[beta_instrument_name.upper()]
    except KeyError:
        raise ValueError("'{}' does not exists in the given DataFrame".format(beta_instrument_name.upper()))
    return _general_dataframe_function(multivariate_df=multivariate_df, func=_realized_beta, return_lag=return_lag, skip_nan=skip_nan,
                                       window=window, beta_instrument_series=beta_instrument_series, beta_return_lag=return_lag,
                                       beta_skip_nan=skip_nan)


def _realized_beta(multivariate_df: pd.DataFrame, func_param: dict):
    """
    When skipping nan, only calculate betas for rows where both series has a non-nan value
    :param multivariate_df: DataFrame
    :param func_param: dict
    :return: DataFrame
    """
    # name the required keys to be used to lookup the parameters
    window_p = 'window'
    beta_instrument_series_p = 'beta_instrument_series'
    beta_return_lag_p = 'beta_return_lag'
    beta_skip_nan_p = 'beta_skip_nan'
    check_required_keys([window_p, beta_instrument_series_p, beta_return_lag_p, beta_skip_nan_p], func_param)

    beta_instrument_series = func_param[beta_instrument_series_p]

    if func_param[beta_skip_nan_p]:
        # drop nan from the beta instrument series and name it
        beta_instrument_series = beta_instrument_series.dropna()
        beta_instrument_series.name = 'beta'
        if func_param[beta_return_lag_p]:
            beta_instrument_series = beta_instrument_series.pct_change(func_param[beta_return_lag_p], fill_method=None)

        # merge the series with the beta instrument series and drop all nan (in order to only compare non nan pairs)
        merged_df = pd.concat([multivariate_df, beta_instrument_series], axis=1)
        merged_df = merged_df.dropna()

        # perform rolling covariance
        ticker = list(merged_df)[0]
        multivariate_df = merged_df[[ticker]]
        beta_instrument_series = merged_df['beta']
    else:
        if func_param[beta_return_lag_p]:
            beta_instrument_series = beta_instrument_series.pct_change(func_param[beta_return_lag_p], fill_method=None)
    # calculate the beta as the ratio between the covariance and variance
    covariance = get_rolling_object(multivariate_df=multivariate_df, window=func_param[window_p]).cov(beta_instrument_series)
    variance = get_rolling_object(multivariate_df=beta_instrument_series, window=func_param[window_p]).var()
    return covariance.divide(variance, axis='index')


def get_rolling_object(multivariate_df: {pd.DataFrame, pd.Series}, window: {int, None}):
    """
    Returns a Rolling object where the window can either be fixed or an increasing one from the start
    :param multivariate_df: pd.DataFrame or pd.Series
    :param window: int
    :return: Rolling
    """
    adj_window = window if window else multivariate_df.shape[0]
    min_periods = window if window else 1
    return multivariate_df.rolling(window=adj_window, min_periods=min_periods)


def check_required_keys(required_keys: list, dictionary: dict):
    """
    Raises an error if any of the elements in the given list does not exists as a key in the given dictionary
    :param required_keys: list
    :param dictionary: dict
    :return: None
    """
    if any(req_key not in dictionary.keys() for req_key in required_keys):
        raise ValueError("'%s' are not specified" % "', '".join(set(required_keys).difference(dictionary.keys())))
    return


def _general_dataframe_function(multivariate_df: pd.DataFrame, func, return_lag: {int, None}, skip_nan: bool, **func_param):
    """
    Applies a function to a DataFrame
    :param multivariate_df: pd.DataFrame
    :param func: any function that takes a DataFrame and a dictionary with parameters
    :param return_lag: int
    :param skip_nan: bool if True, loop through each column and calculate the function after dropping all nan
    :param func_param: dict
    :return: pd.DataFrame
    """
    if skip_nan:
        # initialize a DataFrame used to store the result
        total_result = pd.DataFrame(index=multivariate_df.index)

        # TODO find the tickers that only has non-nan or starts with some nan and afterwards only has non-nan values
        clean_tickers = []
        dirty_tickers = []

        # loop through columns, remove nans, calculate returns (if applicable), apply the function and join the result
        for col_name in list(multivariate_df):
            clean_series = multivariate_df[col_name].dropna()
            if return_lag:
                clean_series = clean_series.pct_change(return_lag, fill_method=None)
            # apply the function and join series to the result
            total_result = total_result.join(
                func(
                    clean_series,
                    func_param
                )
            )
        return total_result
    else:
        if return_lag:
            multivariate_df = multivariate_df.pct_change(return_lag, fill_method=None)
        return func(multivariate_df, func_param)


def main():
    from excel_tools import load_df
    close_df = load_df(full_path=r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\excel_data\data_requests\raw_etf_close_30_nov_2020.xlsx',
                       sheet_name='close_price')
    # close_df = close_df[list(close_df)[:2]]
    # close_df.fillna(method='ffill', inplace=True)


    print('Calculate stat ...')
    stat = holding_period_return(multivariate_df=close_df, lag=100, skip_nan=True, ending_lag=20)

    print(stat)
    stat.to_clipboard()


if __name__ == '__main__':
    main()





