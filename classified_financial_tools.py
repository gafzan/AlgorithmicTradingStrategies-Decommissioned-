"""
classified_financial_tools.py
"""
import pandas as pd
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt

import logging

# my modules
from financial_database import FinancialDatabase
from config_database import my_database_name
from general_tools import progression_bar_str

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def clean_data(multivariate_data: pd.DataFrame):
    """
    Rolls all the N/A forward with the latest available value and then drops all of the N/A (i.e. all columns will start
    at the same index position.
    :param multivariate_data: DataFrame
    :return: DataFrame
    """
    logger.debug('Clean the multivariate data for N/A.')
    multivariate_data.fillna(method='ffill', inplace=True)  # roll all N/A forward
    multivariate_data.dropna(inplace=True)  # remove rows with N/A
    return multivariate_data


def calculate_levered_strategy(strategy_returns: pd.DataFrame, volatility_target: float)->pd.DataFrame:
    """
    Adjusts the strategy returns with the ratio of a certain volatility target and realized volatility.
    :param strategy_returns: DataFrame
    :param volatility_target: float
    :return: DataFrame
    """
    logger.debug('Apply a {} volatility target.'.format(str(100 * volatility_target) + '%'))
    realized_volatility = strategy_returns.rolling(strategy_returns.shape[0], min_periods=1).std()
    exposure = volatility_target / realized_volatility  # volatility target / realized volatility
    exposure.iloc[0, :] = 0
    exposure = exposure.replace(np.inf, 0)
    levered_strategy_returns = strategy_returns * exposure.shift().values
    levered_strategy_returns.iloc[0, :] = 0
    return levered_strategy_returns


def calculate_arithmetic_return(multivariate_data: pd.DataFrame, return_lag: int = 1):
    """
    Returns a DataFrame containign the arithmetic returns P[t] / P[t-1] - 1 with returns at t=0 set to 0.
    :param multivariate_data: DataFrame
    :param return_lag: int (default = 1)
    :return: DataFrame
    """
    logger.debug('Calculate the arithmetic returns.')
    arithmetic_returns = multivariate_data.pct_change(return_lag)
    arithmetic_returns.iloc[0, :] = 0
    return arithmetic_returns


def calculate_strategy_return(multivariate_price_data: pd.DataFrame, strategy_weights: pd.DataFrame,
                              strategy_suffix: str, volatility_target: float = None):
    """
    Returns a new DataFrame with the weighted returns: R_t * W_[t-1]
    :param multivariate_price_data: DataFrame
    :param strategy_weights: DataFrame
    :param strategy_suffix: str to be added to the column names of multivariate_price_data
    :param volatility_target: float
    :return: DataFrame
    """
    logger.debug('Calculate the strategy returns (weighted returns).')
    if not multivariate_price_data.index.equals(strategy_weights.index):
        raise ValueError('The index for multivariate_price_data {} and strategy_weights {} are not the same.'.format(multivariate_price_data.shape, strategy_weights.shape))

    # calculate the strategy returns
    arithmetic_returns = calculate_arithmetic_return(multivariate_price_data)
    strategy_returns = arithmetic_returns * strategy_weights.shift().values
    strategy_returns.iloc[0, :] = 0
    strategy_returns.columns = [col_name + '_{}_STRATEGY_RETURN'.format(strategy_suffix) for col_name in list(arithmetic_returns)]

    if volatility_target is not None:
        strategy_returns = calculate_levered_strategy(strategy_returns, volatility_target)
    return strategy_returns


def general_smoothing(multivariate_data: pd.DataFrame, lambda_1: float, lambda_2: float)->(pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Computes a trend line, its slope and the deviation of the original time series and its smoothed trend line.
    :param multivariate_data: DataFrame containing the original time series.
    :param lambda_1: float
    :param lambda_2: float
    :return: tuple containing 3 DataFrames: smoothed data, slope and deviation
    """
    logger.debug('Perform general smoothing on a multivariate data set {}.'.format(multivariate_data.shape))
    # compute the smoothed time series
    smoothed_data = pd.DataFrame(index=multivariate_data.index, columns=[col_name + '_SMOOTHED' for col_name in list(multivariate_data)])
    smoothed_data.iloc[:2, :] = multivariate_data.iloc[:2, :].values
    smoothed_data.iloc[2, :] = 1 / (1 + lambda_1) * (multivariate_data.iloc[2, :].values + 2 * lambda_1 * smoothed_data.iloc[1, :].values - lambda_1 * smoothed_data.iloc[0, :].values)
    weight_factor = [1.0, 2 * (lambda_1 + lambda_2), -(lambda_1 + 2 * lambda_2), lambda_2]
    weight_factor = [w / (1 + lambda_1 + lambda_2) for w in weight_factor]
    for t_i in range(3, smoothed_data.shape[0]):
        smoothed_data.iloc[t_i, :] = weight_factor[0] * multivariate_data.iloc[t_i, :].values \
                                     + weight_factor[1] * smoothed_data.iloc[t_i - 1, :].values \
                                     + weight_factor[2] * smoothed_data.iloc[t_i - 2, :].values \
                                     + weight_factor[3] * smoothed_data.iloc[t_i - 3, :].values

    # compute the slope
    slope_data = smoothed_data.diff()
    slope_data.iloc[0, :] = 0
    slope_data.columns = [col_name + '_SLOPE' for col_name in list(multivariate_data)]

    # compute the deviation of the original data from the smoothed data
    deviation = multivariate_data - smoothed_data.values
    deviation.columns = [col_name + '_DEVIATION' for col_name in list(multivariate_data)]
    return smoothed_data, slope_data, deviation


def general_smoothing_drawdown_protection(multivariate_data: pd.DataFrame, lambda_1: float, lambda_2: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Compute a strategy that has a long position as long as the slope of a smoothed trend line is positive (else
    allocation is set to zero). Returns a tuple with 2 DataFrames: weights and cumulative strategy returns.
    :param multivariate_data: DataFrame containing the original time series
    :param lambda_1: float
    :param lambda_2: float
    :return: tuple containing 2 DataFrames: weights and cumulative strategy returns
    """
    logger.debug('Calculate the weights and back test for the drawdown protection strategy.')
    # calculate a position signal: 1 if slope of smoothed time series is positive, else 0
    slope = general_smoothing(multivariate_data, lambda_1, lambda_2)[1]
    position_signal = slope.copy()
    position_signal[position_signal > 0] = 1
    position_signal[position_signal <= 0] = 0
    position_signal.columns = [col_name + '_DDP_SIGNAL' for col_name in list(multivariate_data)]

    # calculate the strategy
    strategy_returns = calculate_strategy_return(multivariate_data, position_signal, 'DDP')
    cumulative_returns = (1.0 + strategy_returns).cumprod()  # (1 + r_1) x (1 + r_2) ...
    cumulative_returns.columns = [col_name + '_DDP_STRATEGY' for col_name in list(multivariate_data)]
    return position_signal, cumulative_returns


def triple_filter_signal(multivariate_data: pd.DataFrame, lambda_1_1: float, lambda_1_2: float, lambda_2_1: float,
                         lambda_2_2: float, lambda_3_1: float, lambda_3_2: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Returns a tuple of three DataFrames: weights of a trend following / momentum strategy, normalized slope and
    deviation of smoothed trend line.
    :param multivariate_data: DataFrame
    :param lambda_1_1: float (1st smoothing)
    :param lambda_1_2: float (1st smoothing)
    :param lambda_2_1: float (2nd smoothing)
    :param lambda_2_2: float (2nd smoothing)
    :param lambda_3_1: float (3rd smoothing)
    :param lambda_3_2: float (3rd smoothing)
    :return: tuple (DataFrame, DataFrame, DataFrame)
    """
    logger.debug('Perform triple filter signal on a multivariate data set {}.'.format(multivariate_data.shape))
    # smooth the multivariate data twice
    smoothed_data = general_smoothing(multivariate_data, lambda_1_1, lambda_1_2)[0]
    smoothed_data_2, slope_2, deviation_2 = general_smoothing(smoothed_data, lambda_2_1, lambda_2_2)

    # calculate a preliminary position signal using the slope and deviation from the second smoothing
    sign_slope = pd.DataFrame(data=np.sign(slope_2.values), index=slope_2.index, columns=list(multivariate_data))
    sign_deviation = pd.DataFrame(data=np.sign(deviation_2.values), index=deviation_2.index, columns=list(multivariate_data))
    signal = 0.5 * (sign_slope + sign_deviation)

    # the preliminary signal will be adjusted using the values of a third smoothing
    slope_3, deviation_3 = general_smoothing(multivariate_data, lambda_3_1, lambda_3_2)[1:]
    signal_equals_one = np.logical_and(slope_2.values < 0, deviation_3.values > 0)
    signal[signal_equals_one] = 1.0

    # normalize the third slope and deviation
    slope_tfs = slope_3 / multivariate_data.values
    slope_tfs.columns = [col_name + '_TFS' for col_name in list(slope_3)]
    deviation_tfs = deviation_3 / multivariate_data.values
    deviation_tfs.columns = [col_name + '_TFS' for col_name in list(deviation_3)]
    return signal, slope_tfs, deviation_tfs


def time_series_trend(multivariate_data: pd.DataFrame, lambda_1_1: float, lambda_1_2: float, lambda_2_1: float,
                      lambda_2_2: float, lambda_3_1: float, lambda_3_2: float, volatility_target: float = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns the weights and returns of a trend following strategy. The strategy can be adjusted with a volatility target
    mechanism.
    :param multivariate_data: DataFrame
    :param lambda_1_1: float
    :param lambda_1_2: float
    :param lambda_2_1: float
    :param lambda_2_2: float
    :param lambda_3_1: float
    :param lambda_3_2: float
    :param volatility_target: float
    :return: tuple (DataFrame, DataFrame)
    """
    logger.info('Calculate the weights and back test for the time series trend strategy.')
    # calculate preliminary positions and normalized slopes and deviations using the triple filter signal
    position_tfs, slope_tfs, deviation_tfs = triple_filter_signal(multivariate_data, lambda_1_1, lambda_1_2, lambda_2_1,
                                                                  lambda_2_2, lambda_3_1, lambda_3_2)

    # calculate the cross sectional mean of the magnitude of the normalized slopes from the 'triple filter signal'
    slope_tfs_abs_mean = slope_tfs.abs().mean(axis=1)

    # calculate the position of the time series trend strategy
    adjustment_1 = slope_tfs.iloc[1:, :].div(slope_tfs_abs_mean.iloc[1:], axis=0)
    adjustment_2 = (1 + adjustment_1.pow(2)).pow(-0.5)
    position = position_tfs.copy()
    position.iloc[1:, :] = (1 / multivariate_data.shape[1]) * position_tfs.iloc[1:, :] * adjustment_2.values
    position.columns = [col_name + '_TST_WEIGHT' for col_name in list(multivariate_data)]

    strategy_returns = calculate_strategy_return(multivariate_data, position, 'TST', volatility_target)
    return position, strategy_returns


def long_term_cross_sectional_trend(multivariate_data: pd.DataFrame, lambda_1_1: float, lambda_1_2: float,
                                    lambda_2_1: float, lambda_2_2: float, lambda_3_1: float, lambda_3_2: float,
                                    volatility_target: float = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns the weights and returns of a long-term cross sectional trend following strategy. The strategy can be
    adjusted with a volatility target mechanism.
    :param multivariate_data: DataFrame
    :param lambda_1_1: float
    :param lambda_1_2: float
    :param lambda_2_1: float
    :param lambda_2_2: float
    :param lambda_3_1: float
    :param lambda_3_2: float
    :param volatility_target: float
    :return: tuple (DataFrame, DataFrame)
    """
    logger.info('Calculate the weights and back test for the long-term cross sectional trend strategy.')
    # calculate preliminary positions and normalized slopes and deviations using the triple filter signal
    position_tfs, slope_tfs, deviation_tfs = triple_filter_signal(multivariate_data, lambda_1_1, lambda_1_2, lambda_2_1,
                                                                  lambda_2_2, lambda_3_1, lambda_3_2)

    # calculate the cross sectional mean of the normalized deviation from the 'triple filter signal'
    deviation_tfs_mean = deviation_tfs.mean(axis=1)
    deviation_tfs_minus_mean = deviation_tfs.sub(deviation_tfs_mean, axis=0)
    position = 1 / multivariate_data.shape[1] * position_tfs * np.sign(deviation_tfs_minus_mean.values)
    position.columns = [col_name + '_LTXT_WEIGHT' for col_name in list(multivariate_data)]

    strategy_returns = calculate_strategy_return(multivariate_data, position, 'LTXT', volatility_target)
    return position, strategy_returns


def short_term_cross_sectional_trend(multivariate_data: pd.DataFrame, lambda_1_1: float, lambda_1_2: float,
                                     lambda_2_1: float, lambda_2_2: float, volatility_target: float = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns the weights and returns of a short-term cross sectional trend following strategy. The strategy can be
    adjusted with a volatility target mechanism.
    :param multivariate_data: DataFrame
    :param lambda_1_1: float
    :param lambda_1_2: float
    :param lambda_2_1: float
    :param lambda_2_2: float
    :param volatility_target: float
    :return: tuple (DataFrame, DataFrame)
    """
    logger.info('Calculate the weights and back test for the short-term cross sectional trend strategy.')
    # apply the drawdown protection algorithm
    weight_ddp, strategy_ddp = general_smoothing_drawdown_protection(multivariate_data, lambda_1_1, lambda_1_2)

    # compute the signals based on the drawdown protection strategy
    smoothed_data, slope, deviation = general_smoothing(strategy_ddp, lambda_2_1, lambda_2_2)
    norm_dev_abs = deviation / strategy_ddp.values

    # calculate the cross sectional mean of the magnitude of the normalized deviation
    norm_dev_abs_mean = norm_dev_abs.mean(axis=1)

    # compute the cross-sectional position to identify the instruments to long or short
    norm_dev_abs.sub(norm_dev_abs_mean, axis=0)
    abs_norm_dev_sign_adj = pd.DataFrame(data=np.sign(norm_dev_abs.sub(norm_dev_abs_mean, axis=0).values) * norm_dev_abs.values,
                                         index=multivariate_data.index, columns=list(multivariate_data))
    weight_adj = abs_norm_dev_sign_adj.abs().sum(axis=1)
    preliminary_weight = abs_norm_dev_sign_adj.divide(weight_adj.where(weight_adj != 0, np.nan), axis=0)
    preliminary_weight.replace(np.nan, 0)

    # calculate the final position
    preliminary_weight_abs_sum = preliminary_weight.abs().sum(axis=1)
    position = weight_ddp * preliminary_weight.divide(preliminary_weight_abs_sum.where(weight_adj != 0, np.nan), axis=0).values
    position.columns = [col_name + '_STXT_WEIGHT' for col_name in list(multivariate_data)]

    strategy_returns = calculate_strategy_return(multivariate_data, position, 'STXT', volatility_target)
    return position, strategy_returns


def pairwise_difference(multivariate_data: pd.DataFrame)->pd.DataFrame:
    """
    Calculates the difference for each column pairs on a rolling basis.
    :param multivariate_data: DataFrame
    :return: DataFrame
    """
    logger.debug('Calculate the pairwise difference in returns (used in the cross sectional reversion strategy).')
    # initialize the DataFrame that will store the values
    col_names = [col_name_2 + '_MINUS_' + col_name_1 for col_name_1 in list(multivariate_data) for col_name_2 in
                 list(multivariate_data) if col_name_1 != col_name_2]
    pairwise_difference_result = pd.DataFrame(columns=col_names)
    calendar = multivariate_data.index.values

    # for each pair of underlying instrument compute the pairwise difference in return
    accumulated_progress = 0
    for t_i in range(multivariate_data.shape[0]):
        accumulated_progress += 1 / multivariate_data.shape[0]
        if accumulated_progress > 0.1:
            logger.info(progression_bar_str(t_i, multivariate_data.shape[0]))
            accumulated_progress = 0
        pairwise_difference_i = multivariate_data.iloc[t_i, :].values \
                                       - multivariate_data.iloc[t_i, :].values[:, None]
        np.fill_diagonal(pairwise_difference_i, np.nan)  # set all the dialog values to N/A (easier to remove)
        pairwise_difference_i = pairwise_difference_i.flatten()
        pairwise_difference_i = pairwise_difference_i[~np.isnan(pairwise_difference_i)]  # drop the N/A
        pairwise_difference_result.loc[calendar[t_i]] = pairwise_difference_i  # add the returns differences to a DataFrame
    return pairwise_difference_result


def cross_sectional_reversion(multivariate_data: pd.DataFrame, volatility_target: float = None)->(pd.DataFrame, pd.DataFrame):
    """
    Returns the weights and returns of a mean reversion strategy. The strategy can be adjusted with a volatility target
    mechanism.
    :param multivariate_data: DataFrame
    :param volatility_target: float
    :return: tuple (DataFrame, DataFrame)
    """
    logger.info('Calculate the weights and back test for the cross sectional reversion strategy.')
    # for each pair of underlying instrument compute the pairwise differences in returns
    arithmetic_returns = calculate_arithmetic_return(multivariate_data)
    pairwise_return_difference = pairwise_difference(arithmetic_returns)

    # calculate the rolling standard deviation of each pair of return differences
    standard_deviation = pairwise_return_difference.rolling(pairwise_return_difference.shape[0], min_periods=1).std()

    # assuming the mean is zero, check to see if the return differences are 1 standard deviation from the mean
    omega_1 = (-pairwise_return_difference - standard_deviation.values)
    omega_1[omega_1 > 0] = 1
    omega_1[omega_1 <= 0] = 0
    omega_2 = (pairwise_return_difference - standard_deviation.values)
    omega_2[omega_2 > 0] = 1
    omega_2[omega_2 <= 0] = 0

    # Signal = 1: Return_A - Return_B is outside the norm from below => A: bull, B: bear
    # Signal = -1: Return_A - Return_B is outside the norm from above => A: bear, B: bull
    signal = omega_1 - omega_2.values
    signal.iloc[0, :] = 0

    # calculate the strategy weights
    avg_signal_series_list = []
    for instrument in list(arithmetic_returns):
        # calculate the average signal per instrument
        eligible_columns = [col_signal for col_signal in list(signal) if instrument + '_MINUS' in col_signal]
        avg_signal_per_instrument = signal[eligible_columns].mean(axis=1)
        avg_signal_per_instrument.name = instrument + 'MR_WEIGHT'
        avg_signal_series_list.append(avg_signal_per_instrument)
    strategy_weights = pd.concat(avg_signal_series_list, axis=1)

    # calculate the strategy returns
    strategy_returns = calculate_strategy_return(multivariate_data, strategy_weights, 'MR', volatility_target)

    return strategy_weights, strategy_returns


def retrieve_all_strategies(multivariate_data: pd.DataFrame, lambda_1_1: float, lambda_1_2: float, lambda_2_1: float,
                            lambda_2_2: float, lambda_3_1: float, lambda_3_2: float, volatility_target: float = None):
    """
    Calculates the weights and returns of four algorithmic strategies: time series trend (TST), cross sectional
    reversion (MR), long-term cross sectional trend (LTXT) and short-term cross sectional trend (STXT).
    :param multivariate_data: DataFrame
    :param lambda_1_1: float used in TST, LTXT and STXT
    :param lambda_1_2: float used in TST, LTXT and STXT
    :param lambda_2_1: float used in TST, LTXT and STXT
    :param lambda_2_2: float used in TST, LTXT and STXT
    :param lambda_3_1: float used in TST and LTXT
    :param lambda_3_2: float used in TST and LTXT
    :param volatility_target: float
    :return: DataFrame
    """
    weight_tst, strategy_tst = time_series_trend(multivariate_data, lambda_1_1, lambda_1_2, lambda_2_1, lambda_2_2,
                                                 lambda_3_1, lambda_3_2, volatility_target)
    weight_mr, strategy_mr = cross_sectional_reversion(multivariate_data, volatility_target)
    weight_ltxt, strategy_ltxt = long_term_cross_sectional_trend(multivariate_data, lambda_1_1, lambda_1_2, lambda_2_1,
                                                                 lambda_2_2, lambda_3_1, lambda_3_2, volatility_target)
    weight_stxt, strategy_stxt = short_term_cross_sectional_trend(multivariate_data, lambda_1_1, lambda_1_2, lambda_2_1,
                                                                  lambda_2_2, volatility_target)
    # combine all the DataFrames into two
    weight = pd.concat([weight_tst, weight_mr, weight_ltxt, weight_stxt], axis=1)
    strategy_returns = pd.concat([strategy_tst, strategy_mr, strategy_ltxt, strategy_stxt], axis=1)
    return weight, strategy_returns


def main():
    find_db = FinancialDatabase(my_database_name)
    tickers = ['^OMX', '^GSPC', '^N225']

    # retrieve the data and clean it up
    closing_prices = find_db.get_close_price_df(tickers)  # raw closing prices
    closing_prices.fillna(method='ffill', inplace=True)  # roll all N/A forward
    closing_prices.dropna(inplace=True)  # remove rows with N/A
    weekly_closing_prices = closing_prices[closing_prices.index.weekday == 2]  # only select wednesdays

    # alpha_comb = retrieve_all_strategies(weekly_closing_prices, 20, 20, 20, 20, 20, 20)
    # print(alpha_comb)

    # # perform short-term cross sectional trend
    # strat_ret = short_term_cross_sectional_trend(weekly_closing_prices, 20, 20, 20, 20)[1]
    # cumulative_return = (1 + strat_ret).cumprod()
    # normalized_weekly_closing_prices = weekly_closing_prices.loc[:, :] / weekly_closing_prices.iloc[0, :]
    # combined_df = normalized_weekly_closing_prices.join(cumulative_return)
    # combined_df.plot()
    # plt.show()

    # perform long term cross sectional trend
    # long_term_cross_sectional_trend(weekly_closing_prices, 20, 20, 20, 20, 20, 20)

    # # perform time series trend
    # strat_ret = time_series_trend(weekly_closing_prices, 20, 20, 20, 20, 20, 20)[1]
    # cumulative_return = (1 + strat_ret).cumprod()
    # normalized_weekly_closing_prices = weekly_closing_prices.loc[:, :] / weekly_closing_prices.iloc[0, :]
    # combined_df = normalized_weekly_closing_prices.join(cumulative_return)
    # combined_df.plot()
    # plt.show()

    # # perform triple filter signal
    # weights, slope_tfs, deviation_tfs = triple_filter_signal(weekly_closing_prices, 20, 20, 20, 20, 20, 20)
    # print(weights)
    # print(slope_tfs)
    # print(deviation_tfs)

    # calculate the cross sectional dispersion
    # strat_ret = cross_sectional_reversion(weekly_closing_prices)[1]
    # cumulative_return = (1 + strat_ret).cumprod()
    # normalized_weekly_closing_prices = weekly_closing_prices.loc[:, :] / weekly_closing_prices.iloc[0, :]
    # combined_df = normalized_weekly_closing_prices.join(cumulative_return)
    # combined_df.plot()
    # plt.show()

    # # calculate DDP strategy
    # lambda_1 = 10.0
    # lambda_2 = 50.0
    # cumulative_return = general_smoothing_drawdown_protection(weekly_closing_prices, lambda_1, lambda_2)[1]
    #
    # # plot results
    # normalized_weekly_closing_prices = weekly_closing_prices.loc[:, :] / weekly_closing_prices.iloc[0, :]
    # combined_df = normalized_weekly_closing_prices.join(cumulative_return)
    # combined_df.plot(title='lambda_1 = {}, lambda_2 = {}'.format(lambda_1, lambda_2))
    # plt.show()

    # # perform smoothing
    # lambda_1 = 10.0
    # lambda_2 = 50.0
    # smooth, slope, deviation = general_smoothing(weekly_closing_prices, lambda_1, lambda_2)
    #
    # # plot results
    # combined_df = weekly_closing_prices.join(smooth)
    # combined_df.plot(title='lambda_1 = {}, lambda_2 = {}'.format(lambda_1, lambda_2))
    # plt.show()


if __name__ == '__main__':
    main()

