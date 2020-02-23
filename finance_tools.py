import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm


def realized_volatility(price_df: pd.DataFrame, *vol_lag, annualized_factor: int = 252, allowed_number_na: int = 5) \
        -> pd.DataFrame:
    """Assumes price_df is a DataFrame filled with daily prices as values, tickers as column names and observation dates
    as index. Assumes that measurement_interval and annualized_factor is int and data_availability_threshold is a float.
    Returns a DataFrame with the rolling annualized realized volatility."""
    if min(vol_lag) < 2:
        raise ValueError("vol_lag needs to be an 'int' larger or equal to 2.")
    max_volatility_df = None
    for lag in vol_lag:
        return_df = price_df.pct_change(fill_method=None)
        volatility_sub_df = return_df.rolling(window=lag, min_periods=allowed_number_na).std() \
                            * (annualized_factor ** 0.5)
        if max_volatility_df is None:
            max_volatility_df = volatility_sub_df
        else:
            max_volatility_df = pd.concat([max_volatility_df, volatility_sub_df]).max(level=0, skipna=False)
    # before price starts publishing, value should be nan regardless of data_availability_threshold
    adjustment_df = price_df.pct_change().fillna(method='ffill').rolling(window=max(vol_lag)).mean().isnull()
    adjustment_df = np.where(adjustment_df, np.nan, 1)
    max_volatility_df *= adjustment_df
    return max_volatility_df


def relative_sma(price_df: pd.DataFrame, sma_lag: int, data_availability_threshold: float = 0.9) -> pd.DataFrame:
    """Assumes price_df is a DataFrame filled with daily prices as values, tickers as column names and observation dates
    as index. Assumes that measurement_interval and annualized_factor is int and data_availability_threshold is a float.
    Returns a DataFrame with the simple moving average (SMA) divided by the spot."""
    sma_df = rolling_average(price_df, sma_lag, data_availability_threshold)
    relative_sma_df = sma_df / price_df.fillna(method='ffill')
    return relative_sma_df


def rolling_average(data_df: pd.DataFrame, avg_lag: int, data_availability_threshold: float = 0.9) -> pd.DataFrame:
    """Assumes data_df is a DataFrame filled data as values, tickers as column names and observation dates
    as index. Assumes that measurement_interval and annualized_factor is int and data_availability_threshold is a float.
    Returns a DataFrame with the simple moving average (SMA)."""
    if avg_lag < 1:
        raise ValueError("avg_lag needs to be an 'int' larger or equal to 1")
    minimum_measurement_points = int(data_availability_threshold * avg_lag)
    rolling_avg = data_df.rolling(window=avg_lag, min_periods=minimum_measurement_points).mean()
    # before price starts publishing, value should be nan regardless of data_availability_threshold
    adjustment_df = data_df.fillna(method='ffill').rolling(window=avg_lag).mean().isnull()
    adjustment_df = np.where(adjustment_df, np.nan, 1)
    rolling_avg *= adjustment_df
    return rolling_avg


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


def maximum_drawdown(price_df: pd.DataFrame, look_back_period: int = None) -> pd.Series:
    """Assumes that price_df is a DataFrame and look_back_period is an int. If look_back_period is not assigned, the
    'peak/maximum' will be observed continuously. Returns a Series containing the maximum drawdown for each underlying
    i.e. the lowest 'price' / 'maximum priced over look back period' - 1 observed."""
    drawdown_df = rolling_drawdown(price_df, look_back_period)
    return drawdown_df.min()


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
                             start_date=None, end_date=None) -> dict:
    """Assumes that underlying_price_df is a DataFrame with prices in each column and dates as index, has_benchmark is
    boolean. Calculates annual returns, annual volatility drawdown and return distributions.
    If has_benchmark is True (only works if there are two columns) function will return active return and information
    ratio. Returns a dictionary with names (string) as keys and DataFrames as values. This can later easely be saved in
    an Excel workbook."""
    if has_benchmark and len(underlying_price_df.columns) != 2:
        raise ValueError("Price DataFrame needs to have only two columns: 1st is your strategy and the 2nd is your "
                         "benchmark. Currently the DataFrame has {} column(s).".format(len(underlying_price_df.columns)))
    underlying_price_df.dropna(inplace=True)  # drops each row if there exists a NaN in either column
    if start_date:
        underlying_price_df = underlying_price_df[start_date:]
    if end_date:
        underlying_price_df = underlying_price_df[:end_date]
    performance_df = underlying_price_df / underlying_price_df.iloc[0, :] * 100.0

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
