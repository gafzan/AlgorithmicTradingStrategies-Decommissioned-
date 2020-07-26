import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import seaborn as sns
from dataframe_tools import merge_two_dataframes_as_of


def relative_sma(price_df: pd.DataFrame, sma_lag: int, max_number_of_na: int = 5) -> pd.DataFrame:
    """Assumes price_df is a DataFrame filled with daily prices as values, tickers as column names and observation dates
    as index. Assumes that sma_lag and allowed_number_na are int. Returns a DataFrame with the simple moving average
    (SMA) divided by the spot."""
    sma_df = rolling_average(price_df, sma_lag, max_number_of_na)
    relative_sma_df = sma_df / price_df.fillna(method='ffill')
    return relative_sma_df


def realized_volatility(multivariate_price_df: pd.DataFrame = None, multivariate_return_df: pd.DataFrame = None,
                        vol_lag: {int, list, tuple}=60, annualized_factor: int = 252, allowed_number_na: int = 5,
                        aggregate_func_when_multiple_lags: str = 'max') -> pd.DataFrame:
    """
    Calculates the realized volatility of each column.
    :param multivariate_price_df:
    :param multivariate_return_df:
    :param vol_lag:
    :param annualized_factor:
    :param allowed_number_na:
    :param aggregate_func_when_multiple_lags:
    :return:
    """
    if multivariate_price_df is None and multivariate_return_df is None:
        raise ValueError('Need to specify multivariate_return_df or multivariate_price_df.')
    elif multivariate_return_df is not None and multivariate_price_df is not None:
        raise ValueError('Can only specify one of multivariate_return_df and multivariate_price_df.')

    return _rolling_calc(multivariate_df=multivariate_price_df if multivariate_return_df is None else multivariate_return_df, lag_parameter=vol_lag,
                         convert_to_returns=multivariate_return_df is None,
                         function='realized_volatility', aggregate_method=aggregate_func_when_multiple_lags,
                         max_number_of_na=allowed_number_na, annualized_factor=annualized_factor,
                         minimum_allowed_lag=2)


def beta(multivariate_price_df: pd.DataFrame = None, multivariate_return_df: pd.DataFrame = None, beta_price_df: pd.DataFrame = None,
         beta_lag: {int, list, tuple}=252, return_lag: int = 1, allowed_number_na: int = 5,
         aggregate_func_when_multiple_lags: str = 'avg') -> pd.DataFrame:
    """
    Calculates the beta of each column with respect to a given price DataFrame.
    :param multivariate_price_df:
    :param multivariate_return_df:
    :param beta_price_df:
    :param beta_lag:
    :param return_lag:
    :param allowed_number_na:
    :param aggregate_func_when_multiple_lags:
    :return:
    """
    if multivariate_price_df is None and multivariate_return_df is None:
        raise ValueError('Need to specify multivariate_return_df or multivariate_price_df.')
    elif multivariate_return_df is not None and multivariate_price_df is not None:
        raise ValueError('Can only specify one of multivariate_return_df and multivariate_price_df.')
    return _rolling_calc(
        multivariate_df=multivariate_price_df if multivariate_return_df is None else multivariate_return_df,
        price_data_for_beta_calc_df=beta_price_df,
        lag_parameter=beta_lag, return_lag=return_lag,
        convert_to_returns=multivariate_return_df is None,
        function='beta', aggregate_method=aggregate_func_when_multiple_lags,
        max_number_of_na=allowed_number_na,
        minimum_allowed_lag=2)


def rolling_average(data_df: pd.DataFrame, avg_lag: int, max_number_of_na: {int, None}=5) -> pd.DataFrame:
    """
    Calculates a rolling average. If nan, value is rolled forward except when number of consecutive nan exceeds
    max_number_of_na.
    :param data_df: pandas.DataFrame
    :param avg_lag: int
    :param max_number_of_na: int (default None i.e. all nan are rolled forward)
    :return: pandas.DataFrame
    """
    return _rolling_calc(data_df, avg_lag, False, 'average', max_number_of_na=max_number_of_na, minimum_allowed_lag=1)


def _rolling_calc(multivariate_df: pd.DataFrame, lag_parameter: {int, tuple, list}, convert_to_returns: bool, function: str,
                  aggregate_method: str = None, max_number_of_na: int = 5, return_lag: int = 1,
                  annualized_factor: int = 252, price_data_for_beta_calc_df: pd.DataFrame = None, minimum_allowed_lag: int = 2):
    lag_parameter = _parameter_input_check(lag_parameter, minimum_allowed_lag)
    if multivariate_df.shape[0] < max(lag_parameter) + convert_to_returns + return_lag - 1:
        raise ValueError('multivariate_df needs to have at least {} rows.'.format(max(lag_parameter) + convert_to_returns + return_lag - 1))
    col_list = multivariate_df.columns[multivariate_df.iloc[1:, :].isna().any()].tolist()
    col_with_only_values = multivariate_df.columns[~multivariate_df.iloc[1:, :].isna().any()].tolist()
    col_list.append(col_with_only_values)
    result_df = None
    for lag in lag_parameter:
        result_sub_df = pd.DataFrame(index=multivariate_df.index)
        for col_name in col_list:
            if not isinstance(col_name, list):
                col_name = [col_name]
                df_clean = multivariate_df.loc[:, col_name]
                df_clean.iloc[1:, :] = df_clean.iloc[1:, :].dropna()
            else:
                df_clean = multivariate_df.loc[:, col_name]
            if convert_to_returns:
                df_clean = df_clean.pct_change(return_lag)
            # here is where the main calculation is done
            df_clean = _function_calc(df_clean, function, lag=lag, return_lag=return_lag,
                                      price_data_for_beta_calc_df=price_data_for_beta_calc_df,
                                      annualized_factor=annualized_factor)
            result_sub_df = result_sub_df.join(df_clean)
        result_sub_df = result_sub_df[list(multivariate_df)]
        if result_df is None:
            result_df = result_sub_df
        elif aggregate_method.lower() in ['max', 'maximum']:
            result_df = pd.concat([result_df, result_sub_df]).max(level=0, skipna=False)
        elif aggregate_method.lower() in ['mean', 'average', 'avg']:
            result_df = pd.concat([result_df, result_sub_df]).mean(level=0, skipna=False)
        else:
            if aggregate_method is None:
                raise ValueError("Need to specify aggregate_method when specifying a list of lag parameters.")
            else:
                raise ValueError("'{}' is not a recognized aggregation function.".format(aggregate_method.lower()))

    result_df.fillna(method='ffill', inplace=True)
    result_df = _set_nan_for_missing_data(multivariate_df, result_df, max_number_of_na=max_number_of_na)
    return result_df


def _function_calc(df: pd.DataFrame, func_name: str, **kwargs):
    if func_name == 'realized_volatility':
        df_clean = df.rolling(window=kwargs['lag']).std() * (kwargs['annualized_factor'] ** 0.5)
    elif func_name == 'average':
        df_clean = df.rolling(window=kwargs['lag']).mean()
    elif func_name == 'beta':
        if kwargs['price_data_for_beta_calc_df'] is None:
            raise ValueError('Need to specify data_for_beta_calc_df when calculating betas.')
        else:
            df_clean = merge_two_dataframes_as_of(df, kwargs['price_data_for_beta_calc_df'], 'used_for_beta_calc')
        df_clean.iloc[:, -1] = df_clean.iloc[:, -1].pct_change(kwargs['return_lag'])
        covariance_df = df_clean.rolling(window=kwargs['lag']).cov(df_clean.iloc[:, -1])
        variance_df = df_clean.iloc[:, -1].rolling(window=kwargs['lag']).var()
        df_clean = covariance_df.divide(variance_df, axis='index')
        df_clean = df_clean.iloc[:, :-1]  # ignore the column furthest to the right
    else:
        raise ValueError("'{}' is not a recognized function.".format(func_name.lower()))
    return df_clean


def _parameter_input_check(param: {int, tuple, list}, minimum_value: int):
    """
    Checks and converts the parameter to a list if necessary.
    :param param: int, list
    :param minimum_value: int
    :return: None
    """
    # convert to list if applicable
    try:
        param[0]
    except TypeError:
        param = [param]
    # check value
    if min(param) < minimum_value:
        raise ValueError('Parameter value needs to be greater or equal to {}.'.format(minimum_value))
    return param


def _set_nan_for_missing_data(original_multivariate_df: pd.DataFrame, calculated_value_df: pd.DataFrame,
                              max_number_of_na: int)->pd.DataFrame:
    """
    After calculating e.g. Volatility, this script sets the values to NaN if the original DataFrame had a number of
    consecutive rows of NaN above a threshold.
    :param original_multivariate_df: DataFrame
    :param calculated_value_df: DataFrame
    :param max_number_of_na: int
    :return: DataFrame
    """
    # set value to nan if the number of consecutive nan exceeds max_number_of_na
    adjustment_df = original_multivariate_df.rolling(window=max_number_of_na + 1, min_periods=1).mean()
    eligibility_df = np.where(adjustment_df.isna(), np.nan, 1)
    return calculated_value_df * eligibility_df


def rolling_drawdown(price_df: pd.DataFrame, look_back_period: int = None) -> pd.DataFrame:
    """Assumes that price_df is a DataFrame and look_back_period is an int. If look_back_period is not assigned, the
    'peak/maximum' will be observed continuously. Returns a DataFrame containing the drawdown for each underlying i.e.
    'price' / 'maximum priced over look back period' - 1."""
    if look_back_period is None:
        look_back_period = len(price_df.index)
    if look_back_period < 1:
        raise ValueError("look_back_lag' needs to be larger or equal to 1.")
    price_df = price_df.fillna(method='ffill').copy()
    rolling_max_df = price_df.rolling(window=look_back_period, min_periods=1).max()
    drawdown_df = price_df / rolling_max_df - 1.0
    return drawdown_df


def exponentially_weighted_return(price_df: pd.DataFrame, lambda_: float) -> pd.DataFrame:
    """
    Calculates the exponentially weighted returns
    :param price_df: pd.DataFrame containing the closing levels
    :param lambda_: the weight
    :return: pd.DataFrame
    """
    price_return_df = price_df.pct_change()
    price_return_df = price_return_df.iloc[1:, :]
    number_days = price_return_df.shape[0]
    exp_weighting_var = [np.exp(-lambda_ / number_days * (1 + t_day)) for t_day in range(number_days - 1, -1, -1)]
    exp_weighting_s = pd.Series(index=price_return_df.index, data=exp_weighting_var)
    return price_return_df.mul(exp_weighting_s, axis=0)


def maximum_drawdown(price_df: pd.DataFrame, look_back_period: int = None) -> pd.Series:
    """Assumes that price_df is a DataFrame and look_back_period is an int. If look_back_period is not assigned, the
    'peak/maximum' will be observed continuously. Returns a Series containing the maximum drawdown for each underlying
    i.e. the lowest 'price' / 'maximum priced over look back period' - 1 observed."""
    drawdown_df = rolling_drawdown(price_df, look_back_period)
    return drawdown_df.min()


def plot_pairwise_correlation(df: pd.DataFrame):
    """ Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html"""
    sns.set(style="white")
    corr = df.corr()  # Compute the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=np.bool))  # Generate a mask for the upper triangle
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Generate a custom diverging colormap
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def _check_and_merge_price_weight_df(price_df: pd.DataFrame, weight_df: pd.DataFrame) -> pd.DataFrame:
    """(Used by 'index_calculation')
    Assumes price_df and weight_df are both DataFrames. Checks if the two DataFrames have the same column headers and
    returns adjusted price and weight DataFrame to have a valid start date."""
    price_df = price_df.reindex(sorted(price_df.columns), axis=1)  # sort column headers
    weight_df = weight_df.reindex(sorted(weight_df.columns), axis=1)  # sort column headers
    price_df.fillna(method='ffill', inplace=True)  # forward fill using previous price if price is NaN
    if list(price_df) != list(weight_df):
        raise ValueError('The tickers (column headers) of price_df and weight_df are not the same.')
    weight_df['rebalance'] = range(len(weight_df.index))  # counter (indicator) for rebalance date
    price_index_reset_df = price_df.reset_index()
    weight_index_reset_df = weight_df.reset_index()
    left_on_col_name = list(price_index_reset_df)[0]
    right_on_col_name = list(weight_index_reset_df)[0]
    price_weight_df = pd.merge_asof(price_index_reset_df, weight_index_reset_df, left_on=left_on_col_name,
                                    right_on=right_on_col_name,
                                    suffixes=['_price', '_weight'])  # merge the price and weight DataFrames
    price_weight_df.set_index(price_weight_df[left_on_col_name], inplace=True)  # set dates as index
    price_weight_df.drop([left_on_col_name, right_on_col_name], inplace=True, axis=1)  # remove old 'index' column
    weight_column_names = [col_name for col_name in list(price_weight_df) if '_weight' in col_name]
    price_weight_df = price_weight_df.dropna(subset=weight_column_names)  # remove rows where weights are NaN
    price_weight_df['rebalance'] = price_weight_df['rebalance'].diff()  # 1: new weight, else 0
    price_weight_df.iloc[0, -1] = 1
    return price_weight_df


def _calculate_performance(weight_price_df: pd.DataFrame) -> pd.DataFrame:
    """(Used by 'index_calculation')
    Assumes weight_price_df is a DataFrame. Returns a new DataFrame with columns containing the performance."""
    tickers = [col_name.replace('_price', '') for col_name in list(weight_price_df) if '_price' in col_name]
    # Get the price at each rebalance date and then roll the value
    for ticker in tickers:
        weight_price_df[ticker + '_price_last_rbd'] = weight_price_df[ticker + '_price'] * \
                                                      (weight_price_df['rebalance'] == 1)
        weight_price_df[ticker + '_price_last_rbd'].replace(0, np.nan, inplace=True)
        weight_price_df.fillna(method='ffill', inplace=True)  # forward fill

    # Calculate the performance
    performance_col_name = [ticker + '_performance' for ticker in tickers]
    weight_price_df[performance_col_name] = pd.DataFrame(data=weight_price_df.filter(regex='_price$').values /
                                                              weight_price_df.filter(regex='_price_last_rbd$').shift(1).values,
                                                         index=weight_price_df.index)

    # Calculate the weighted performance
    weighted_performance_col_names = [ticker + '_weighted_return' for ticker in tickers]
    weight_price_df[weighted_performance_col_names] = pd.DataFrame(data=weight_price_df.filter(regex='_weight$').shift(1).values * \
                                                                        (weight_price_df.filter(regex='_performance$').values - 1.0),
                                                                   index=weight_price_df.index)
    return weight_price_df


def index_calculation(price_df: pd.DataFrame, weight_df: pd.DataFrame, transaction_cost: float = 0, fee: float = 0,
                      initial_amount: float = 100.0) -> pd.DataFrame:
    """Assumes price_df and weight_df are both DataFrames that have the same column headers, transaction_cost and fee 
    are floats. Returns an index as a DataFrame."""
    if transaction_cost < 0:
        raise ValueError('transaction_cost needs to be equal or greater to 0.')
    weight_price_df = _check_and_merge_price_weight_df(price_df, weight_df)
    index_calculation_df = _calculate_performance(weight_price_df)
    index_calculation_df['transaction_cost'] = index_calculation_df.filter(regex='_weight$').diff().abs().sum(axis=1) \
                                               * transaction_cost
    index_calculation_df['gross_return'] = index_calculation_df.filter(regex='_weighted_return$').sum(axis=1)

    # Calculate the index
    index_calculation_df['index'] = np.nan  # Initialize the column
    index_calendar = index_calculation_df.index
    index_last_rbd = initial_amount
    last_rbd = index_calculation_df.index[0]
    index_calculation_df.loc[last_rbd, 'index'] = index_last_rbd
    for date in index_calendar[1:]:
        accumulated_fee = (date - last_rbd).days / 360.0 * fee
        index_calculation_df.loc[date, 'index'] = index_last_rbd * (1.0 + index_calculation_df.loc[date, 'gross_return']
                                                                    - accumulated_fee
                                                                    - index_calculation_df.loc[date, 'transaction_cost'])
        if index_calculation_df.loc[date, 'rebalance']:  # if it is a rebalance date
            index_last_rbd = index_calculation_df.loc[date, 'index']  # new index since last rebalance date
            last_rbd = date  # new rebalance date
    return index_calculation_df


def index_daily_rebalanced(multivariate_daily_returns: pd.DataFrame, weights: pd.DataFrame,
                           transaction_costs: float = 0, rolling_fee_pa: float = 0, weight_smoothing_lag: int = 1,
                           weight_observation_lag: int = 1, initial_value: float = 100, volatility_target: float = None,
                           volatility_lag: {int, list, tuple}=60, risky_weight_cap: float = 1,
                           market_price_df: pd.DataFrame = None, beta_lag: {int, list, tuple}=252,
                           beta_hedge_carry_cost_pa: float = 0.0, beta_return_lag: int = 1):
    # calculate the gross return of the index
    multivariate_daily_returns.iloc[1:, :].fillna(0, inplace=True)
    num_instruments = multivariate_daily_returns.shape[1]
    index_result = merge_two_dataframes_as_of(multivariate_daily_returns, weights, '_WEIGHT')
    index_result.iloc[:, num_instruments:] = index_result.iloc[:, num_instruments:].rolling(window=weight_smoothing_lag, min_periods=1).mean()  # smooth the weights to reduce turnover
    index_result[[col_name + '_WEIGHTED_RETURN' for col_name in list(multivariate_daily_returns)]] = index_result.iloc[:, :num_instruments] * index_result.iloc[:, num_instruments:].shift(weight_observation_lag).values

    # make the index 'market-neutral' by adding a short position where the allocation depends on the realized beta
    if market_price_df is not None:

        index_result = _add_beta_hedge(index_result=index_result, multivariate_daily_returns=multivariate_daily_returns,
                                       market_price_df=market_price_df, beta_lag=beta_lag,
                                       beta_return_lag=beta_return_lag, num_instruments=num_instruments,
                                       weight_observation_lag=weight_observation_lag)
        market_neutral = True
        num_instruments += 1
    else:
        market_neutral = False

        # index_result = _add_beta_hedge(index_result)
    index_result['GROSS_INDEX_RETURN'] = index_result.iloc[:, 2 * num_instruments:].sum(axis=1, skipna=False)

    # add volatility target mechanism if applicable
    if volatility_target:
        index_result = _add_volatility_target(index_result, volatility_target, volatility_lag, risky_weight_cap, num_instruments,
                                              weight_observation_lag)
    # calculate the gross index
    index_result['GROSS_INDEX'] = initial_value * (1 + index_result['GROSS_INDEX_RETURN']).cumprod()
    start_of_index_i = index_result['GROSS_INDEX_RETURN'].index.get_loc(index_result['GROSS_INDEX_RETURN'].first_valid_index()) - 1
    index_result.iloc[start_of_index_i, -1] = initial_value

    # adjust index for transaction costs and index fees if any
    if transaction_costs != 0 or rolling_fee_pa != 0 or market_neutral * beta_hedge_carry_cost_pa != 0:
        # add a column with the net index return
        index_result = _add_net_index_return(index_result=index_result, transaction_costs=transaction_costs,
                                             rolling_fee_pa=rolling_fee_pa, number_of_instruments=num_instruments,
                                             beta_carry_cost=market_neutral * beta_hedge_carry_cost_pa,
                                             weight_observation_lag=weight_observation_lag)

        # calculate the net index
        index_result['NET_INDEX'] = initial_value * (1 + index_result['NET_INDEX_RETURN']).cumprod()
        index_result.iloc[start_of_index_i, -1] = initial_value
    return index_result


def _add_beta_hedge(index_result: pd.DataFrame, multivariate_daily_returns: pd.DataFrame, market_price_df: pd.DataFrame,
                    beta_lag: {int, list}, beta_return_lag: int, num_instruments: int, weight_observation_lag: int):
    # add the daily returns of the market instrument
    index_result = merge_two_dataframes_as_of(index_result, market_price_df.pct_change(), left_suffix='BETA_INSTRUMENT')

    # calculate the weight of the short position
    if beta_return_lag > 1:
        # convert the daily returns back to performance data
        multivariate_price_df = multivariate_daily_returns.shift(-1)  # shift the daily returns backwards
        multivariate_price_df[~multivariate_price_df.isnull()] = 1  # where there is a shifted return set value to 1
        multivariate_price_df.iloc[-1, :] = 1  # last rows should be 1 since the shifted value is always nan
        multivariate_price_df += multivariate_daily_returns.fillna(0)  # 1 + daily return
        multivariate_price_df = multivariate_price_df.cumprod()  # (1 + R1) * (1 + R2) * ...
        beta_per_stock_df = beta(multivariate_price_df=multivariate_price_df, beta_price_df=market_price_df,
                                 beta_lag=beta_lag, return_lag=beta_return_lag)
    else:
        beta_per_stock_df = beta(multivariate_return_df=multivariate_daily_returns, beta_price_df=market_price_df,
                                 beta_lag=beta_lag)
    # calculate the weighted average across all stock betas
    weighted_beta_df = pd.DataFrame(
        data=(beta_per_stock_df * index_result.iloc[:, num_instruments: 2 * num_instruments].values).sum(axis=1,
                                                                                                         skipna=False),
        columns=['BETA_WEIGHT'])
    weighted_beta_df *= -1  # since you are shorting the market
    index_result = index_result.join(weighted_beta_df)
    # calculate the short beta position
    index_result[list(index_result)[-2] + '_WEIGHTED_RETURN'] = index_result.iloc[:, -2] * \
                                                                index_result.iloc[:,-1]\
                                                                    .shift(weight_observation_lag).values

    # rearrange the columns
    instrument_columns = list(index_result)[:num_instruments]
    instrument_columns.append(list(index_result)[-3])  # add the beta instrument
    weight_columns = list(index_result)[num_instruments: 2 * num_instruments]
    weight_columns.append(list(index_result)[-2])  # add the beta weight
    weighted_return_columns = [col_name for col_name in list(index_result) if col_name.endswith('_WEIGHTED_RETURN')]
    all_col_names = instrument_columns + weight_columns + weighted_return_columns
    index_result = index_result[all_col_names]
    return index_result


def _add_volatility_target(index_result: pd.DataFrame, volatility_target: float, volatility_lag: {int, list},
                           risky_weight_cap: float, number_of_instruments: int, weight_observation_lag: int):
    # calculate the risky weight
    realized_volatility_gross_index = realized_volatility(multivariate_return_df=index_result[['GROSS_INDEX_RETURN']], vol_lag=volatility_lag)
    risky_weight = volatility_target / realized_volatility_gross_index
    risky_weight[risky_weight >= risky_weight_cap] = risky_weight_cap
    index_result = index_result.join(pd.DataFrame(data=risky_weight.values, index=risky_weight.index, columns=['RISKY_WEIGHT']))

    # add new weights columns with weights adjusted based on the risky weight
    weight_post_vt_col_names = [col_name + '_WEIGHT_POST_VT' for col_name in list(index_result)[:number_of_instruments]]
    index_result[weight_post_vt_col_names] = index_result.iloc[:, number_of_instruments: 2 * number_of_instruments].multiply(index_result['RISKY_WEIGHT'], axis=0)

    # adjust the weighted returns and gross index returns
    weighted_return_col_names = [col_name + '_WEIGHTED_RETURN' for col_name in list(index_result)[:number_of_instruments]]
    index_result[weighted_return_col_names] = index_result.iloc[:, :number_of_instruments] * index_result.loc[:, weight_post_vt_col_names].shift(weight_observation_lag).values
    index_result['GROSS_INDEX_RETURN'] = index_result.loc[:, weighted_return_col_names].sum(axis=1, skipna=False)
    return index_result


def _add_net_index_return(index_result: pd.DataFrame, transaction_costs: float, rolling_fee_pa: float,
                          number_of_instruments: int, beta_carry_cost: float, weight_observation_lag: int):
    # calculate the index net of transaction costs
    weight_col_names = [col_name for col_name in list(index_result) if col_name.endswith('_WEIGHT_POST_VT')]
    if not len(weight_col_names):
        weight_col_names = list(index_result)[number_of_instruments: 2 * number_of_instruments]

    abs_weight_delta = index_result[weight_col_names].diff().abs().sum(axis=1)
    index_result['NET_INDEX_RETURN'] = index_result['GROSS_INDEX_RETURN'] - transaction_costs * abs_weight_delta.values

    # calculate the index net of index fees
    dt = [None] + [(index_result.index[n] - index_result.index[n - 1]).days / 365 for n in
                   range(1, len(index_result.index))]
    dt_s = pd.Series(data=dt, index=index_result.index)
    index_result['NET_INDEX_RETURN'] -= rolling_fee_pa * dt_s

    # calculate the index net of the rolling cost of carrying the short beta position
    index_result['NET_INDEX_RETURN'] += index_result['BETA_WEIGHT'].shift(weight_observation_lag) * dt_s.values * beta_carry_cost

    return index_result


def monthly_return_table(price_df: pd.DataFrame, include_first_monthly_return: bool = True) -> {list, pd.DataFrame}:
    """Assumes price_df is a DataFrame. Returns a DataFrame with monthly returns and yearly returns. If price_df has
    more than one column, script returns a list filled with DataFrames."""
    month_name_dict = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8,
                       'Oct': 9, 'Nov': 10, 'Dec': 11}  # dictionary used to name the columns in the table
    price_df = price_df.copy()
    tickers = list(price_df)
    price_df['Month'] = price_df.index.strftime('%b')
    price_df['Year'] = price_df.index.strftime('%Y')

    # within each month, roll forward if there is nan
    clean_price_df = price_df.groupby(['Month', 'Year'], as_index=False).apply(lambda group: group.ffill())
    monthly_price_df = clean_price_df.resample('BM').apply(lambda x: x[-1])  # look at last day of each month

    # calculate the returns only for the price columns and not month and year columns
    monthly_return_df = monthly_price_df.apply(lambda x: x.pct_change() if x.name in tickers else x)

    if include_first_monthly_return:  # calculate the return of the first 'broken' month
        first_available_price_df = clean_price_df.groupby(['Month', 'Year'], as_index=False).apply(lambda group: group.bfill()).resample('BM').apply(lambda x: x[0])
        for ticker in tickers:
            # this looks messy but it works :)
            first_available_value = first_available_price_df.loc[first_available_price_df[ticker].notna().idxmax(),
                                                                 ticker]
            first_end_of_month_index_pos = monthly_return_df.index.get_loc(monthly_return_df[ticker].notna().idxmax())
            first_available_end_of_month_value = monthly_price_df.iloc[first_end_of_month_index_pos - 1,
                                                                       tickers.index(ticker)]
            first_month_return = first_available_end_of_month_value / first_available_value - 1.0
            monthly_return_df.iloc[first_end_of_month_index_pos - 1, tickers.index(ticker)] = first_month_return

    column_names = list(set(month_name_dict.keys()) & set(monthly_return_df['Month'].values))  # intersection
    column_names.sort(key=lambda name: month_name_dict[name])  # sort the columns in monthly order

    if len(tickers) == 1:  # only one underlying
        monthly_return_table_df = monthly_return_df.pivot(index='Year', columns='Month', values=tickers[0])
        monthly_return_table_df = monthly_return_table_df[column_names]  # correct the column order
        monthly_return_table_df['Yearly return'] = monthly_return_table_df.add(1.0).fillna(1.0).cumprod(axis=1).iloc[:, -1] - 1.0
        return monthly_return_table_df
    else:
        monthly_return_table_df_list = []
        for ticker in tickers:
            monthly_return_table_df = monthly_return_df.pivot(index='Year', columns='Month', values=ticker)
            monthly_return_table_df = monthly_return_table_df[column_names]  # correct the column order
            monthly_return_table_df['Yearly return'] = monthly_return_table_df.add(1.0).fillna(1.0).cumprod(axis=1).iloc[:, -1] - 1.0
            monthly_return_table_df_list.append(monthly_return_table_df)
        return monthly_return_table_df_list


def return_and_risk_analysis(underlying_price_df: pd.DataFrame, has_benchmark=False, print_results=True,
                             start_date=None, end_date=None, normalize: bool = True) -> dict:
    """Assumes that underlying_price_df is a DataFrame with prices in each column and dates as index, has_benchmark is
    boolean. Calculates annual returns, annual volatility drawdown and return distributions.
    If has_benchmark is True (only works if there are two columns) function will return active return and information
    ratio. Returns a dictionary with names (string) as keys and DataFrames as values. This can later easely be saved in
    an Excel workbook."""
    if has_benchmark and len(underlying_price_df.columns) != 2:
        raise ValueError("Price DataFrame needs to have only two columns: 1st is your strategy and the 2nd is your "
                         "benchmark. Currently the DataFrame has {} column(s).".format(len(underlying_price_df.columns)))
    underlying_price_df = underlying_price_df.copy()
    underlying_price_df.dropna(inplace=True)  # drops each row if there exists a NaN in either column
    if start_date:
        underlying_price_df = underlying_price_df[start_date:]
    if end_date:
        underlying_price_df = underlying_price_df[:end_date]

    if normalize:
        performance_df = underlying_price_df / underlying_price_df.iloc[0, :] * 100.0
    else:
        performance_df = underlying_price_df

    # annual return
    rolling_1y_return_df = underlying_price_df.pct_change(252).dropna()
    avg_1y_return_s = rolling_1y_return_df.mean()

    # annual volatility
    rolling_1y_volatility_df = underlying_price_df.pct_change().rolling(window=252).std().dropna() * math.sqrt(252)
    avg_1y_volatility_s = rolling_1y_volatility_df.mean()

    # sharpe ratio
    sharpe_ratio_s = avg_1y_return_s / avg_1y_volatility_s

    # maximum drawdown
    rolling_1y_drawdown_df = rolling_drawdown(underlying_price_df, 252)
    max_drawdown_s = maximum_drawdown(underlying_price_df)

    # combine the Series into a DataFrame
    risk_return_table_df = pd.concat([avg_1y_return_s, avg_1y_volatility_s, sharpe_ratio_s, max_drawdown_s],
                                     axis=1).transpose()

    # setup a dictionary with sheet names as keys and DataFrames as values
    sheet_name_df_dict = {'Performance': performance_df, 'Risk and return': risk_return_table_df,
                          'Rolling 1Y return': rolling_1y_return_df, 'Rolling 1Y volatility': rolling_1y_volatility_df,
                          'Rolling 1Y drawdown': rolling_1y_drawdown_df}

    if has_benchmark:
        # calculate the rolling active return and information ratio
        rolling_1y_active_return_s = rolling_1y_return_df.iloc[:, 0] - rolling_1y_return_df.iloc[:, 1]
        rolling_standard_error_s = rolling_1y_active_return_s.rolling(window=252).std()
        rolling_1y_information_ratio_s = (rolling_1y_active_return_s / rolling_standard_error_s).dropna()
        avg_1y_active_return_df = pd.DataFrame(data=[[rolling_1y_active_return_s.mean(), '--']],
                                               columns=list(underlying_price_df))
        avg_1y_information_ratio_df = pd.DataFrame(data=[[rolling_1y_information_ratio_s.mean(), '--']],
                                                   columns=list(underlying_price_df))
        # add active returns and information ratio to the table DataFrame
        risk_return_table_df = pd.concat([risk_return_table_df, avg_1y_active_return_df, avg_1y_information_ratio_df])
        risk_return_table_df.index = ['Average 1Y return', 'Average 1Y volatility', 'Sharpe ratio', 'Maximum drawdown',
                                      'Average 1Y active return', 'Average information ratio']
        # include active returns and information ratio DataFrames in the dictionary
        sheet_name_df_dict.update \
                (
                    {'Rolling 1Y active returns': pd.DataFrame({'1Y Active return': rolling_1y_active_return_s}),
                        'Rolling 1Y information ratio': pd.DataFrame({'Information ratio': rolling_1y_information_ratio_s})
                     }
                )
    else:
        risk_return_table_df.index = ['Average 1Y return', 'Average 1Y volatility', 'Sharpe ratio', 'Maximum drawdown']
    sheet_name_df_dict['Risk and return'] = risk_return_table_df

    # add the monthly return tables
    monthly_and_yearly_return_table = monthly_return_table(performance_df)
    if not isinstance(monthly_and_yearly_return_table, list):
        monthly_and_yearly_return_table = [monthly_and_yearly_return_table]
    for i in range(len(performance_df.columns)):
        sheet_name_df_dict.update\
            (
                {'Return table (' + list(performance_df)[i] + ')': monthly_and_yearly_return_table[i]}
            )

    if print_results:
        print('\n' + 100 * '-' + '\nRisk and return')
        print(risk_return_table_df)
        print('\n' + 100 * '-' + '\n1Y return')
        print(rolling_1y_return_df.describe())
        print('\n' + 100 * '-' + '\n1Y volatility')
        print(rolling_1y_volatility_df.describe())
        print('\n' + 100 * '-' + '\n1Y drawdown')
        print(rolling_1y_drawdown_df.describe())
        start_date_str = str(underlying_price_df.index[0])[:10]
        end_date_str = str(underlying_price_df.index[-1])[:10]
        print('Source: Yahoo Finance and Huggorm Investment AB. Based on data between {} and {}. Past performance is '
              'not a reliable indicator of future returns.'.format(start_date_str, end_date_str))
    return sheet_name_df_dict


def plot_results(df_dict: {dict}):
    """Assumes that df_dict is a dictionary with strings as keys and DataFrames as values. This dictionary is assumed to
    have been generated using the return_and_risk_analysis function. Plots a selection of the DataFrames."""
    cmap = cm.get_cmap('Dark2')  # background color
    plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme
    try:
        # this DataFrame is only available if you have chosen to calculate results compared with a benchmark
        df_dict['Rolling 1Y active returns'].plot(grid=True, title='1Y active return', colormap=cmap)
    except KeyError:
        pass
    performance_df = df_dict['Performance']
    df_dict['Rolling 1Y return'].plot(kind='hist', grid=True, subplots=True, sharex=True, sharey=True,
                                      title='1Y return distribution\n({} - {})'.format(str(performance_df.index[0])[:10],
                                                                                       str(performance_df.index[-1])[:10]),
                                      colormap=cmap, bins=100)
    df_dict['Rolling 1Y drawdown'].plot(grid=True, title='1Y drawdown', colormap=cmap)
    df_dict['Rolling 1Y volatility'].plot(grid=True, title='1Y volatility', colormap=cmap)
    df_dict['Rolling 1Y return'].plot(grid=True, title='1Y return', colormap=cmap)
    performance_df.plot(grid=True, title='Performance', colormap=cmap)
    plt.show()


def main():
    from excel_tools import load_df
    full_path = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\excel_data\price_test.xlsx'
    price = load_df(full_path=full_path, sheet_name='price')
    price = price.pct_change()
    beta_price = load_df(full_path=full_path, sheet_name='beta')
    weight = load_df(full_path=full_path, sheet_name='weight')
    print('calc...')
    # b = beta(price, beta_price_df=beta_price, beta_lag=20)
    index_df = index_daily_rebalanced(price, weight, market_price_df=beta_price, beta_lag=20, beta_return_lag=3,
                                      beta_hedge_carry_cost_pa=0.005)
    index_df.to_clipboard()


if __name__ == '__main__':
    main()
