"""
rolling_futures_index
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
from matplotlib import pyplot as plt

import logging

from sqlalchemy import and_, or_

# my modules
from database.financial_database import FinancialDatabase
from database.models_db import Underlying
from database.config_database import my_database_name

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

__MONTH_CODE__ = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
__MONTH__ = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep',
                       10: 'Oct', 11: 'Nov', 12: 'Dec'}

# source: https://data.bloomberglp.com/indices/sites/2/2018/02/BCOM-Methodology-January-2018_FINAL-2.pdf
__COMMODITY_BBG_UNDERLYING_CODE__ = {"CL": "WTI Crude Oil", "CO": "Brent Crude Oil", "HO": "ULS Diesel",
                                     "XB": "RBOB Gasoline", "NG": "Natural Gas", "SI": "Silver", "GC": "Gold",
                                     "PL": "Platinum", "HG": "Copper", "LA": "Aluminum", "LN": "Nickel", "LX": "Zinc",
                                     "LL": "Lead", "LT": "Tin", "LC": "Live Cattle", "LH": "Lean Hogs",
                                     "BO": "Soybean Oil", "SM": "Soybean Meal", "S ": "Soybeans", "C ": "Corn",
                                     "W ": "Chicago Wheat", "KW": "KC HRW Wheat", "CC": "Cocoa", "CT": "Cotton",
                                     "SB": "Sugar", "KC": "Coffee"}

# source: https://data.bloomberglp.com/indices/sites/2/2018/02/BCOM-Methodology-January-2018_FINAL-2.pdf
__ROLLING_MONTHS_COMMODITY_CODE__ = {"CL": ["H", "K", "N", "U", "X", "F"],
                                     "CO": ["H", "K", "N", "U", "X", "F"],
                                     "HO": ["H", "K", "N", "U", "X", "F"],
                                     "XB": ["H", "K", "N", "U", "X", "F"],
                                     "NG": ["H", "K", "N", "U", "X", "F"],
                                     "SI": ["H", "K", "N", "U", "Z"],
                                     "GC": ["G", "J", "M", "Q", "Z"],
                                     "PL": ["J", "N", "V", "F"],
                                     "HG": ["H", "K", "N", "U", "Z", "F"],
                                     "LA": ["H", "K", "N", "U", "X", "F"],
                                     "LN": ["H", "K", "N", "U", "X", "F"],
                                     "LX": ["H", "K", "N", "U", "X", "F"],
                                     "LL": ["H", "K", "N", "U", "X", "F"],
                                     "LT": ["H", "K", "N", "U", "X", "F"],
                                     "LC": ["G", "J", "M", "Q", "V", "Z"],
                                     "LH": ["G", "J", "M", "N", "Q", "V", "Z"],
                                     "BO": ["H", "K", "N", "Z", "F"],
                                     "SM": ["H", "K", "N", "Z", "F"],
                                     "S ": ["H", "K", "N", "X", "F"],
                                     "C ": ["H", "K", "N", "U", "Z"],
                                     "W ": ["H", "K", "N", "U", "Z"],
                                     "KW": ["H", "K", "N", "U", "Z"],
                                     "CC": ["H", "K", "N", "U", "Z"],
                                     "CT": ["H", "K", "N", "Z"],
                                     "SB": ["H", "K", "N", "V"],
                                     "KC": ["H", "K", "N", "U", "Z"]}


def get_investment_universe(underlying_code: str, month_codes: list, start_date: datetime = None, end_date: datetime = None) -> list:
    """
    Returns a list of futures tickers based on the given monthly codes and underlying code
    :param underlying_code: str e.g. 'GC' gives gold futures
    :param month_codes: list of strings e.g. 'M' corresponds to june and 'Q' to august
    :param start_date: datetime only picks contracts where the latest observation date is after this date
    :param end_date: datetime only picks contracts where the oldest observation date is before this date
    :return: list of strings
    """
    # handle inputs
    if start_date is None:
        start_date = datetime(1950, 1, 1)
    if end_date is None:
        end_date = datetime.today() + BDay(1)  # next business day from today

    # combine the underlying code (e.g. 'GC' for gold) and the month codes (e.g. 'M' and 'Q')
    contract_codes = [underlying_code.upper() + month_code.upper() for month_code in month_codes]

    # find eligible futures contracts in the database by filtering based on dates and contract codes
    fin_db = FinancialDatabase(my_database_name)
    query_futures_tickers = fin_db.session.query(
        Underlying.ticker
    ).filter(
        and_(
            or_(
                *[Underlying.ticker.like(contract_code + '%') for contract_code in contract_codes]
            ),
            Underlying.latest_observation_date_with_values > start_date,
            Underlying.oldest_observation_date < end_date,
            Underlying.underlying_type == 'FUTURE'
        )
    ).order_by(
        Underlying.description.asc()
    ).all()

    # store the resulting tickers in a list
    futures_tickers = [tup[0] for tup in query_futures_tickers]
    if not len(futures_tickers):
        raise ValueError('No tickers could be found in the database.\nDatabase: {}\nUnderlying code: {}\nMonth code: %s'
                         .format(my_database_name, underlying_code.upper()) % ', '.join(month_codes))
    return futures_tickers


def get_expiry_date_dict(tickers: list) -> dict:
    """
    Return a dictionary: keys = tickers (str), values = expiry dates (datetime)
    :param tickers: list of strings
    :return: dictionary
    """
    fin_db = FinancialDatabase(my_database_name)
    asset_class = fin_db.get_ticker_underlying_attribute_dict(tickers, Underlying.underlying_type)
    desc = fin_db.get_ticker_underlying_attribute_dict(tickers, Underlying.description)
    result_dict = {}
    for ticker in tickers:
        if asset_class[ticker] == 'FUTURE':
            expiry_desc = desc[ticker]
            result_dict.update({ticker: datetime.strptime(expiry_desc.split()[1], '%Y-%m-%d')})
        else:
            logger.warning('{} is of type {} and does not have a expiry date.'.format(ticker, asset_class[ticker]))
    return result_dict


def get_rolling_calendar_df(tickers: list, roll_period_days: int = 5, expiry_buffer: int = 1) -> pd.DataFrame:
    """
    Returns a DataFrame with the expected rolling start date, rolling end date and expiry date
    :param tickers: list of strings
    :param roll_period_days: int. Number of days during which the contracts are rolled over
    :param expiry_buffer: int days before the expiry date which becomes the rolling end date
    :return: DataFrame
    """
    # check the inputs
    if roll_period_days < 1:
        raise ValueError('roll_period_days needs to be an integer greater or equal to 1.')
    if expiry_buffer < 0:
        raise ValueError('expiry_buffer needs to be an integer greater or equal to 0.')

    # get the expiry dates for all the tickers and then adjust this date to get the rolling calendar
    expiry_date_dict = get_expiry_date_dict(tickers)
    rolling_calendar_df = pd.DataFrame(index=tickers)
    rolling_calendar_df['expiry_date'] = rolling_calendar_df.index.map(expiry_date_dict)
    rolling_calendar_df['rolling_end_date'] = rolling_calendar_df['expiry_date'] - BDay(expiry_buffer)
    rolling_calendar_df['rolling_start_date'] = rolling_calendar_df['rolling_end_date'] - BDay(roll_period_days)
    rolling_calendar_df.sort_values(by=['expiry_date'], inplace=True)  # sort the expiry dates in ascending order
    return rolling_calendar_df[['rolling_start_date', 'rolling_end_date', 'expiry_date']]


def calculate_rolling_future_index(tickers: list, start_date: datetime = None, end_date: datetime = None,
                                   allocation: {float, dict}=1.0, roll_period_days: int = 5, expiry_buffer_days: int = 1,
                                   transaction_costs: float = 0.0, fee_pa: float = 0.0) -> pd.DataFrame:
    """
    Calculate a rolling futures strategy. Rolling method is governed by roll_period_days and expiry_buffer and the
    expiry calendar for the respective tickers. Results can be net of transaction costs and fee p.a.
    :param tickers: list of strings
    :param start_date: datetime
    :param end_date: datetime
    :param allocation: float or dict. if float, the rolling strategy is allocated to the nearest future with weight equal to
    'allocation'. Dictionary values should be {i (int): weight_i (float)} where i represents the ith nearest future and
    weight_i represents the weight of the ith nearest future. A 50%/50% allocation between the 1st and 3rd future would
    correspond to allocation = {1: 0.5, 3: 0.5}
    :param roll_period_days: int number of days where the position is rolled into the next future
    :param expiry_buffer_days: int number of days before the expiry date to end the rolling of the position
    :param transaction_costs: float multiplied by the absolute change in weight
    :param fee_pa: float subtracted from the gross return multiplied by the day count fraction
    :return: DataFrame
    """
    # handle given allocation
    if isinstance(allocation, dict):
        future_weight_dict = allocation
    elif isinstance(allocation, float):
        future_weight_dict = {1: allocation}
    else:
        raise ValueError('allocation needs to be a dictionary {i (int): weight_i (float)} or a float.')

    # get the closing prices for the given futures
    fut_data = get_futures_daily_close_data(tickers, start_date, end_date)

    # get the expiry calendar
    target_rolling_calendar = get_rolling_calendar_df(tickers, roll_period_days, expiry_buffer_days)

    # list of all the positions accross the term structure. E.g. [1, 4] means that the strategy allocates to the 1st and
    # 4th future
    term_structure_position_list = list(future_weight_dict.keys())

    # calculate the daily returns for the futures
    daily_obs_calendar = fut_data.index  # daily observation dates (DatetimeIndex)
    prev__date = daily_obs_calendar.values[0]  # initialize to first observation date
    future_price_return_df = None
    last_roll = False
    for expiry_date in target_rolling_calendar['expiry_date'].values:  # loop through all expiry dates
        if expiry_date < daily_obs_calendar.values[-1]:  # normal roll
            expiry_position_in_daily_calendar = daily_obs_calendar.get_loc(expiry_date)
            roll_end_date = daily_obs_calendar[expiry_position_in_daily_calendar - expiry_buffer_days]
        else:
            last_roll = True
            roll_end_date = daily_obs_calendar.values[-1]  # last observation date
        future_price_df = None  # initialize the price DataFrame
        for term_structure_position in term_structure_position_list:  # loop through all positions accross the term structure
            # select the relevant closing prices
            active_tickers = target_rolling_calendar[target_rolling_calendar['expiry_date'] >= expiry_date].index.values
            relevant_tickers = active_tickers[term_structure_position - 1: term_structure_position + 1]
            sub_future_price_df = fut_data.loc[prev__date:roll_end_date, relevant_tickers]  # select relevant prices
            sub_future_price_df.columns = ['FUT_{}'.format(term_structure_position), 'CONTRACT_AFTER_FUT_{}'.format(term_structure_position)]

            # add weight columns
            weight = future_weight_dict[term_structure_position]
            first_weight_col_name = 'FUT_{}'.format(term_structure_position) + '_WEIGHT'
            sub_future_price_df[first_weight_col_name] = weight

            # adjust the weight during the rolling period
            if last_roll:
                # for the last roll we need to approximate the latest eights based on the expected rolling period
                bday_shift = BDay(roll_period_days + expiry_buffer_days - 1)
                expected_roll_start_date = expiry_date.astype('M8[D]').astype('O') - bday_shift
                sub_future_price_df.loc[expected_roll_start_date:, first_weight_col_name] = 0  # to be smoothed out later
            else:
                sub_future_price_df.iloc[-roll_period_days:, 2] = 0  # to be smoothed out later

            # smooth the weights
            sub_future_price_df[first_weight_col_name] = sub_future_price_df[first_weight_col_name].rolling(roll_period_days).mean()
            sub_future_price_df[first_weight_col_name] = sub_future_price_df[first_weight_col_name].fillna(method='backfill')

            # calculate the weights of the following futures contract
            sub_future_price_df['CONTRACT_AFTER_FUT_{}'.format(term_structure_position) + '_WEIGHT'] = weight - sub_future_price_df.loc[:, first_weight_col_name]

            # concatenate the results (add new columns)
            if future_price_df is None:
                future_price_df = sub_future_price_df
            else:
                future_price_df = pd.concat([future_price_df, sub_future_price_df], join='outer', axis=1)
        # calculate the return of the price columns
        sub_future_price_return_df = future_price_df.copy()
        columns_with_price = [col for col in list(sub_future_price_return_df) if not col.endswith('_WEIGHT')]
        sub_future_price_return_df.loc[:, columns_with_price] = sub_future_price_return_df.loc[:, columns_with_price].pct_change()

        # concatenate the results (add new rows)
        if future_price_return_df is None:
            future_price_return_df = sub_future_price_return_df
        else:
            future_price_return_df = pd.concat([future_price_return_df, sub_future_price_return_df.dropna()],
                                               ignore_index=False, sort=False)

        if last_roll:
            break
        else:
            prev__date = roll_end_date

    # calculate the rolling futures index
    future_price_return_df['INDEX_RETURN'] = 0
    for term_structure_position in term_structure_position_list:
        daily_return_values = future_price_return_df.filter(regex='FUT_{}$'.format(term_structure_position)).values
        daily_weight_values = future_price_return_df.filter(regex='FUT_{}_WEIGHT$'.format(term_structure_position)).shift().values  # shift the weights forward one day
        daily_weighted_return_df = pd.DataFrame(data=daily_return_values * daily_weight_values, index=future_price_return_df.index)
        future_price_return_df['INDEX_RETURN'] += daily_weighted_return_df.sum(axis=1)  # sum of Weight_i x Return_i
    future_price_return_df['INDEX'] = 1 + future_price_return_df.loc[:, 'INDEX_RETURN']  # 1 + daily returns
    future_price_return_df['INDEX'] = future_price_return_df['INDEX'].cumprod()  # compounded weigthed daily returns

    # calculate index net of transaction costs ...
    daily_abs_weight_change_df = 2 * future_price_return_df.filter(regex='(^CONTRACT_AFTER_FUT_).+(_WEIGHT$)').abs().diff().clip(lower=0).sum(axis=1)
    future_price_return_df['NET_INDEX_RETURN'] = future_price_return_df['INDEX_RETURN'] - transaction_costs * daily_abs_weight_change_df

    # ... and rolling fee p.a.
    dt = [None] + [(future_price_return_df.index[n] - future_price_return_df.index[n-1]).days / 365 for n in range(1, len(future_price_return_df.index))]
    dt_s = pd.Series(data=dt, index=future_price_return_df.index)
    future_price_return_df['NET_INDEX_RETURN'] -= fee_pa * dt_s
    future_price_return_df['NET_INDEX'] = 1 + future_price_return_df.loc[:, 'NET_INDEX_RETURN']  # 1 + net daily returns
    future_price_return_df['NET_INDEX'] = future_price_return_df['NET_INDEX'].cumprod()  # compounded daily returns
    return future_price_return_df


def get_futures_daily_close_data(tickers: list, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Returns a DataFrame with the daily close of futures between start and end dates. N/A are rolled forward if they
    exists before the last observation date of the resp. futures contract.
    :param tickers: list of strings
    :param start_date: datetime
    :param end_date: datetime
    :return: DataFrame
    """
    # get the raw data from the financial database
    fin_db = FinancialDatabase(my_database_name)
    raw_futures_data = fin_db.get_close_price_df(tickers, start_date, end_date)

    # clean the data by rolling N/A forward
    cleaned_futures_data = raw_futures_data.fillna(method='ffill')

    # get last observation date per ticker
    ticker_last_obs_date_dict = fin_db.get_ticker_underlying_attribute_dict(tickers, Underlying.latest_observation_date_with_values)

    # loop through each column and set each row to N/A if it is after the last observation date for the resp. ticker
    for col_i in range(cleaned_futures_data.shape[1]):
        last_obs_date = ticker_last_obs_date_dict[list(cleaned_futures_data)[col_i]]
        try:
            last_obs_date_index = cleaned_futures_data.index.get_loc(last_obs_date)
        except KeyError:  # in case when the last observation period is after end_date
            last_obs_date_index = cleaned_futures_data.shape[0]
        cleaned_futures_data.iloc[last_obs_date_index + 1:, col_i] = np.nan
    return cleaned_futures_data


def main():
    underlying_code = 'gc'
    month_codes = [__MONTH_CODE__[2], __MONTH_CODE__[4], __MONTH_CODE__[6], __MONTH_CODE__[8], __MONTH_CODE__[10],
                   __MONTH_CODE__[12]]
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2020, 2, 1)
    futures_tickers = get_investment_universe(underlying_code, month_codes, start_date, end_date)

    rolling_futures_index = calculate_rolling_future_index(futures_tickers, start_date, end_date, roll_period_days=15,
                                                           transaction_costs=0.002, fee_pa=0.005)

    rolling_futures_index[['INDEX', 'NET_INDEX']].plot()
    plt.show()


if __name__ == '__main__':
    main()










