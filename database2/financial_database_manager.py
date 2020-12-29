"""
financial_database_manager.py
"""
import numpy as np
import pandas as pd
import pandas.core.common as com
from pandas.tseries.offsets import BDay
from datetime import date

import logging

from sqlalchemy import func, or_, and_

# my data base configurations
from database2.base import session_factory
from database2.domain_model import Instrument, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, Split
from database2.domain_model import data_table_instrument_types_mapper, available_data_sources, init_instrument_arg_names, instrument_default_value_map
from database2.yahoo_finance import YahooFinanceConnection as _YahooFinanceConnection

# other tools
from tools.general_tools import user_picks_element_from_list, list_grouper, time_period_logger_msg
from tools.dataframe_tools import nan_before_dates, dataframe_with_new_calendar

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FinancialDatabaseManager:

    def __init__(self):
        self.session = session_factory()

    def ticker_existence(self, tickers: {str, list}, data_source: str = None) -> tuple:
        """
        Returns a tuple with three lists: 1) list of existing tickers 2) list of missing tickers 3) list of existing
        tickers but with a different data source than the one provided. If data_source is not specified, only two lists
        will be returned
        :param tickers: str or list
        :param data_source: str
        :return: (list, list, list)
        """
        tickers = make_ticker_list(tickers)
        logger.debug('Checking existence of {} ticker(s)'.format(len(tickers)))
        # filter out the tickers and associated data sources
        q_existing_tickers = self.session.query(
            Instrument.ticker,
            Instrument.data_source) \
            .filter(
            Instrument.ticker.in_(tickers)
        ).all()
        self.session.close()

        # create the resulting ticker lists
        existing_tickers = [q_tup[0] for q_tup in q_existing_tickers]
        missing_tickers = list(set(tickers).difference(existing_tickers))
        if data_source:
            existing_tickers_diff_source = [q_tup[0] for q_tup in q_existing_tickers if q_tup[1] != data_source.upper()]
            return existing_tickers, missing_tickers, existing_tickers_diff_source
        else:
            return existing_tickers, missing_tickers

    def get_ticker_instrument_attribute_dict(self, tickers: {str, list}, attribute: {str, list}) -> dict:
        """
        Assumes that ticker is a string or a list of strings and attribute is a string (e.g. Instrument.sector will
        return a dictionary like {'ticker 1': 'sector A', 'ticker 2': 'sector B' ,...}).
        Returns a dictionary with tickers as keys and the specific attribute as values
        :param tickers:
        :param attribute: str
        :return: dict
        """
        logger.debug('Loads {} from the data base'.format("'%s'" % "', '".join(attribute) if isinstance(attribute, list) else attribute))
        ticker_attribute_dict = {}  # initializing the dictionary
        attributes = [getattr(Instrument, inst_attr) for inst_attr in attribute] if isinstance(attribute, list) \
            else [getattr(Instrument, attribute)]
        # to make the requests smaller, we need to split the ticker list into sub list
        for ticker_sub_list in list_grouper(make_ticker_list(tickers), 500):

            query_ticker_attribute = self.session.query(Instrument.ticker, *attributes) \
                .filter(
                Instrument.ticker.in_(ticker_sub_list)) \
                .order_by(Instrument.ticker)
            if isinstance(attribute, list):
                ticker_attribute_dict.update({tup[0]: dict(zip(attribute, tup[1:])) for tup in query_ticker_attribute.all()})
            else:
                ticker_attribute_dict.update(dict(query_ticker_attribute))
        self.session.close()
        return ticker_attribute_dict

    def get_latest_as_of_date_dict(self, tickers: {str, list},
                                   table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}) -> dict:
        """
        Returns a dictionary with tickers as keys and the latest date with a value in the given data table as values
        :param tickers: str, list
        :param table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}
        :return: dict
        """
        # make the initial query for instrument id and maximum of the dates to later be filtered and grouped by
        latest_as_of_date_ticker_map = {}
        q_instrument_id_max_date = self.session.query(
            Instrument.ticker,
            func.max(table.as_of_date)
        ).join(Instrument)
        for ticker_sub_list in list_grouper(make_ticker_list(tickers), 500):
            # first query the latest observation date for the close price
            sub_q_latest_close_date = q_instrument_id_max_date \
                .filter(
                Instrument.ticker.in_(ticker_sub_list),
            ).group_by(
                table.instrument_id
            )
            latest_as_of_date_ticker_map.update(sub_q_latest_close_date)
        self.session.close()
        return latest_as_of_date_ticker_map

    def get_eligible_start_date(self, tickers: {str, list},
                                data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, list}) -> {date, None}:
        """
        For a given number of tickers and a data table, gather the latest date with a value in the data table and return the
        oldest of these dates. If one ticker does not have a row in the data table, return None.
        :param tickers: str, list
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, list
        if list call the function recursively and return the oldest date for each data table
        :return: None, date
        """

        if isinstance(data_table, list):
            start_dates = []
            for d_tab in data_table:
                # call the function recursively
                start_dates.append(self.get_eligible_start_date(tickers=tickers, data_table=d_tab))
            if None in start_dates:
                return None
            else:
                return min(start_dates)
        else:
            latest_dates = self.get_latest_as_of_date_dict(tickers=tickers, table=data_table).values()
            num_tickers = len(tickers) if isinstance(tickers, list) else 1
            if len(latest_dates) < num_tickers:  # meaning at least one ticker does not have a row in the data table
                return None
            else:
                return min(latest_dates)

    def delete_instrument(self, tickers: {str, list}) -> None:
        """
        Deletes all instruments with the given ticker(s) from the database
        :param tickers: str, list
        :return: None
        """
        tickers = make_ticker_list(tickers)
        q_instrument = self.session.query(Instrument) \
            .filter(
            Instrument.ticker.in_(tickers)
        )
        for instrument in q_instrument:
            logger.debug('Deletes {} from the instrument table'.format(instrument.ticker))
            self.session.delete(instrument)
        self.session.commit()  # triggers a final flush of the remaining changes
        self.session.close()

    # TODO change the inputs to ticker list, start_date and end_date
    def _delete_rows_in_data_table_overlapping_df(self, df: pd.DataFrame, data_table: {OpenPrice, HighPrice, LowPrice,
                                                                                       ClosePrice, Volume, Dividend, Split}):
        """
        Deletes the rows in the given data table class corresponding to overlaps in the given DataFrame
        :param df: DataFrame (columns = instrument tickers, index = DatetimeIndex)
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: None
        """

        # create a list of all the instrument ids corresponding to the tickers
        # we can't join(Instrument) and then filter on Instrument.ticker since delete will raise an error
        instrument_ids = [inst_id for inst_id in
                          self.get_ticker_instrument_attribute_dict(tickers=list(df), attribute='id').values()]

        # delete all data table rows for each relevant tickers with a date between the DataFrame index
        start_date = min(df.index)
        end_date = max(df.index)
        logger.debug("Delete rows in '{}' table for {} ticker(s)".format(data_table.__tablename__, df.shape[1]) + time_period_logger_msg(start_date=start_date, end_date=end_date))

        for sub_instrument_id_list in list_grouper(instrument_ids, 500):
            self.session.query(data_table) \
                .filter(
                and_(
                    data_table.instrument_id.in_(sub_instrument_id_list),
                    and_(data_table.as_of_date >= start_date, data_table.as_of_date <= end_date)
                )
            ).delete(synchronize_session=False)  # don’t synchronize the session. This option is the most efficient and
            # is reliable once the session is expired, which typically occurs after a commit(), or explicitly using
            # expire_all()
        self.session.commit()
        self.session.close()

    def _get_data(self, tickers: {str, list}, start_date: {date, None}, end_date: {date, None},
                  data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, Split}):
        """
        Returns a DataFrame with tickers as columns, as of date as index and data as values (e.g. dividends)
        :param tickers: str, list
        :param start_date: date, None
        :param end_date: date, None
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, Split
        :return: DataFrame
        """
        logger.debug("Load data from the '{}' table".format(data_table.__tablename__) + time_period_logger_msg(start_date=start_date, end_date=end_date))
        sql_table_df = None  # initialize the resulting DataFrame table
        for sub_ticker_list in list_grouper(make_ticker_list(ticker=tickers), 500):
            # make the database query based on tickers, start date and end date
            q_data = self.session.query(
                data_table.as_of_date,
                data_table.value,
                Instrument.ticker)\
                .join(Instrument)\
                .filter(
                and_(
                    Instrument.ticker.in_(sub_ticker_list),
                    data_table.as_of_date >= start_date if start_date is not None else True,
                    data_table.as_of_date <= end_date if end_date is not None else True
                )
            )
            # store data in a DataFrame and concatenate the results
            sql_query_df = pd.read_sql(q_data.statement, q_data.session.bind)
            if sql_table_df is None:
                sql_table_df = sql_query_df
            else:
                sql_table_df = pd.concat([sql_table_df, sql_query_df])

        # pivot the DataFrame
        logger.debug('Pivot DataFrame')
        result_df = pd.pivot_table(sql_table_df, values='value', index=['as_of_date'], columns=['ticker'])
        return result_df

    def _calculate_total_return_price(self, price_df: pd.DataFrame, div_tax: float):
        """
        Adjust the price to take into account the reinvestment of dividends
        :param price_df: DataFrame
        :param div_tax: float
        :return: DataFrame
        """
        logger.debug("Calculate total return for {} ticker(s)".format(price_df.shape[1]))
        if div_tax < 0:
            raise ValueError("div_tax needs to ba a float larger or equal to 0")

        ticker_type_dict = self.get_ticker_instrument_attribute_dict(tickers=list(price_df),
                                                                     attribute='instrument_type')
        # find the tickers that has a dividend paying instrument type and download the data
        div_paying_ticker = [v for v, k in ticker_type_dict.items() if k in data_table_instrument_types_mapper[Dividend]]
        dividend_df = self.get_dividend(tickers=div_paying_ticker, start_date=min(price_df.index), end_date=max(price_df.index))
        dividend_df = dividend_df.dropna(axis=1, how='all')
        dividend_df = dividend_df.fillna(value=0) * (1.0 - div_tax)
        div_paying_ticker = list(dividend_df)

        # adjust the price data by adding the dividend (after tax), calculating the cumulative daily total return and
        div_pay_price_df = price_df[div_paying_ticker].copy()
        div_pay_price_clean_df = div_pay_price_df.fillna(method='ffill')
        div_yield_df = dividend_df.loc[div_pay_price_clean_df.index].divide(div_pay_price_clean_df.shift())
        daily_tr = div_pay_price_clean_df.pct_change() + div_yield_df.fillna(value=0).values
        cum_tr = (1 + daily_tr.fillna(value=0)).cumprod()

        # set the initial non nan price to the original one and remove total return price where the original was nan
        index_first_non_nan = div_pay_price_clean_df.notna().idxmax()  # index of first non-NaN for each column
        first_value = np.diag(div_pay_price_clean_df.loc[index_first_non_nan])  # get the first non-NaN for each column
        cum_tr *= first_value  # to have the same initial value as the original DataFrame
        nan_or_1 = div_pay_price_df.copy()
        nan_or_1[~nan_or_1.isnull()] = 1
        cum_tr *= nan_or_1.values

        price_df[div_paying_ticker] = cum_tr
        return price_df

    def _convert_fx(self, df: pd.DataFrame, fx: str):
        pass

    def _get_price(self, tickers: {str, list}, start_date: {date, None}, end_date: {date, None},
                   data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice}, total_return: bool, div_tax: float):
        price_df = self._get_data(tickers=tickers, start_date=start_date, end_date=end_date, data_table=data_table)
        if total_return:
            return self._calculate_total_return_price(price_df=price_df, div_tax=div_tax)
        else:
            return price_df

    def get_dividend(self, tickers: {str, list}, start_date: date = None, end_date: date = None):
        return self._get_data(tickers=tickers, start_date=start_date, end_date=end_date, data_table=Dividend)

    def get_volume(self, tickers: {str, list}, start_date: date = None, end_date: date = None):
        return self._get_data(tickers=tickers, start_date=start_date, end_date=end_date, data_table=Volume)

    def get_close_price(self, tickers: {str, list}, start_date: date = None, end_date: date = None,
                        total_return: bool = False, div_tax: float = 0, fx: str = None):
        return self._get_price(tickers=tickers, start_date=start_date, end_date=end_date, data_table=ClosePrice,
                               total_return=total_return, div_tax=div_tax)


class _DatabaseFeeder(FinancialDatabaseManager):

    _new_old_key_map = {}
    _old_new_value_per_old_key_map = {}
    _data_tables = None

    def __init__(self, source: str = None, as_of_date: date = None):
        super().__init__()
        # set the data source (can only be set once and is read only)
        if source is None:
            self._source = source
        elif source.upper() in available_data_sources:
            self._source = source.upper()
        else:
            raise ValueError("'source' needs to be one of '%s'" % "', '".join(available_data_sources))
        self.as_of_date = as_of_date

    def _find_unrefreshed_tickers(self, tickers: {str, list},
                                  data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, list}):
        """
        Takes a given number of tickers and a data table object and filters away the tickers that does not support the given
        data table (e.g. a ticker representing a FX rate would be filtered away if the given data table object was Dividend),
        and checks the latest recorded refresh date. Filter away tickers that has a recorded refresh date set to today
        :param tickers: str, list
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: list
        """
        unrefreshed_tickers = []
        if isinstance(data_table, list):
            for d_tab in data_table:
                unrefreshed_tickers.extend(self._find_unrefreshed_tickers(tickers=tickers, data_table=d_tab))
            return list(set(unrefreshed_tickers))
        else:
            logger.debug("Looking for tickers with unrefreshed data in the '{}' table".format(data_table.__tablename__))
            q_ticker = self.session.query(Instrument.ticker)  # initial query for the tickers
            for tickers_sub_list in list_grouper(make_ticker_list(tickers), 500):
                sub_q_ticker = q_ticker \
                    .filter(
                    and_(
                        Instrument.ticker.in_(tickers_sub_list),  # tickers should be in the sub list
                        Instrument.instrument_type.in_(data_table_instrument_types_mapper[data_table]),  # the instrument type is eligible
                        or_(
                            # the ticker has no data in the table
                            getattr(Instrument, data_table.refresh_info_column_name) == None,
                            # the latest refresh was in the past
                            getattr(Instrument, data_table.refresh_info_column_name) <= self.as_of_date
                        )
                    )
                ).all()
                unrefreshed_tickers.extend([q_tick[0] for q_tick in sub_q_ticker])
            self.session.close()
            return unrefreshed_tickers

    def _download_instrument_info_dict(self, tickers: list):
        raise ValueError("have not defined method that loads instrument information")

    def _download_data_dict(self, tickers: list, start_date: {date, None}, end_date: {date, None}) -> dict:
        raise ValueError("have not defined method that loads instrument information")

    def _info_dict_translator(self, info_dict: dict)->dict:
        """
        Takes a ticker information dictionary (key=tickers, values = sub-dictionary with information) and changes the
        keys and values in the sub-dictionrary based on the translator attributes '_new_old_key_translator' and
        '_old_new_value_translator_per_key'
        :param info_dict: dict
        :return: dict
        """
        logger.debug('Translate the tickers and values of the information dictionary')
        # for each ticker, the keys and values of the given dictionary according to the specified mappers
        adj_info = {}
        for ticker, info_per_ticker in info_dict.items():
            # find the keys where the corresponding value needs to change
            value_adj_keys = [key for key in self._old_new_value_per_old_key_map.keys() if key in info_per_ticker.keys()]

            # change each value according to the mapper
            for value_adj_key in value_adj_keys:
                info_per_ticker.update(
                    {
                        value_adj_key: self._old_new_value_per_old_key_map[value_adj_key].get(
                            info_per_ticker[value_adj_key],
                            info_per_ticker[value_adj_key]
                        )
                    }
                )

            # change each key according to the mapper (when there is no key in the old dict set a default value to the
            # key in the new dictionary)
            adj_info_per_ticker = {key: info_per_ticker.get(key,
                                                            info_per_ticker.get(
                                                                self._new_old_key_map.get(
                                                                    key,
                                                                    key
                                                                ),
                                                                instrument_default_value_map[key]
                                                            )
                                                            ) for key in init_instrument_arg_names}
            adj_info_per_ticker.update({'data_source': self.source})  # add the data source
            adj_info.update(
                {
                    ticker: adj_info_per_ticker
                }
            )
        logger.debug('Done with adjusting the information dictionary')
        return adj_info

    def add_instrument(self, tickers) -> None:
        """

        :param tickers:
        :return:
        """
        existence = self.ticker_existence(tickers=tickers, data_source=self.source)
        missing_tickers = existence[1]
        existing_tickers_diff_source = existence[2]
        if len(existing_tickers_diff_source):
            tasks = ['Delete & Replace', 'Replace', 'Refresh', 'Ignore']
            print('There exists {} ticker(s) in the database with a different data source than {}\nWhat should we do?\n'
                  .format(len(existing_tickers_diff_source), self.source))
            chosen_task = user_picks_element_from_list(list_=tasks)
            if chosen_task == tasks[0]:  # Delete & Replace
                # delete the instruments and add the tickers as missing
                self.delete_instrument(existing_tickers_diff_source)
                missing_tickers.extend(missing_tickers)
            elif chosen_task == tasks[1]:  # Replace
                # keep the instruments in the table but change the attributes and set the dates to None
                # load the daily data and remove all data in the database that overlaps with this data
                info_dict = self._download_instrument_info_dict(tickers=existing_tickers_diff_source)
                adj_info_dict = self._info_dict_translator(info_dict=info_dict)
                self.adjusting_instrument_attributes(ticker_attributes_dict=adj_info_dict)
                self.refresh_data(tickers=existing_tickers_diff_source, replace=True, use_eligible_start_date=False)
            elif chosen_task == tasks[2]:  # Refresh
                # load the daily data (based on the latest date in the database) and remove all loaded daily data that
                # overlaps with the database
                self.refresh_data(tickers=existing_tickers_diff_source)
            elif chosen_task == tasks[3]:
                # do nothing
                pass
            else:
                raise ValueError(f"'{chosen_task}' is not a recognized task")

        if len(missing_tickers):
            logger.debug('Add {} tickers to the database'.format(len(missing_tickers)))
            for sub_ticker_list in list_grouper(missing_tickers, 10):
                info_dict = self._download_instrument_info_dict(tickers=sub_ticker_list)
                adj_info_dict = self._info_dict_translator(info_dict=info_dict)
                self._populate_instrument_table(ticker_attributes_dict=adj_info_dict)
            # self.refresh_data(tickers=missing_tickers)
        else:
            logger.info("All tickers already exists in the database")

    def refresh_data(self, tickers: {str, list}, replace: bool = False, use_eligible_start_date: bool = True,
                     data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, list} = None):
        """

        :param tickers: str, list
        :param replace: bool
        :param use_eligible_start_date: bool
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, list
        :return:
        """

        if data_table is None:
            data_table = self._data_tables
        else:
            data_table = data_table if isinstance(data_table, list) else [data_table]
            if any(d_tab not in self._data_tables for d_tab in data_table):
                raise ValueError("{} can only refresh data for the following tables: '%s'".format(type(self).__name__) % "', '".join(self._data_tables))

        # check if some tickers are missing from the data base and raise an error if that is the case
        missing_tickers = self.ticker_existence(tickers=tickers)[1]
        if len(missing_tickers):
            raise ValueError("{} tickers are missing from the database:\n'%s'".format(len(missing_tickers)) % "', '".join(missing_tickers))

        # remove rows from the data tables where the 'as of date' is equal to the 'created by' date
        # this is to always have historical data after the close
        self._delete_data_rows_before_close(tickers=tickers, data_table=data_table)

        tickers = self._find_unrefreshed_tickers(tickers=tickers, data_table=data_table)
        if len(tickers) == 0:
            logger.debug("All tickers are refreshed")
            return

        if use_eligible_start_date:
            # get the correct start date
            start_date = self.get_eligible_start_date(tickers=tickers, data_table=self._data_tables)
        else:
            start_date = None
        logger.debug("Refresh data for {} ticker(s)".format(len(tickers)) + time_period_logger_msg(start_date=start_date, end_date=None))
        # download the DataFrame dictionary and populate the data table rows
        table_name_data_dict = self._download_data_dict(tickers=tickers, start_date=start_date, end_date=date.today())
        self._populate_data_table(table_df_dict={table: table_name_data_dict[table]
                                                 for table in table_name_data_dict.keys() & data_table}, replace=replace)

    def _delete_data_rows_before_close(self, tickers: {str, list}, data_table: list)->None:
        """
        Deletes the data table rows for the given tickers where the 'as of date' == 'created by' date. This is to avoid
        downloading and storing data that has been loaded before the close
        :param tickers: str, list
        :return: None
        """
        instrument_id_list = self.get_ticker_instrument_attribute_dict(tickers=tickers, attribute='id').keys()
        for d_tab in data_table:
            self.session.query(
                d_tab
            ).filter(
                d_tab.instrument_id.in_(instrument_id_list),
                d_tab.as_of_date == d_tab.created_at
            ).delete(synchronize_session=False)
        self.session.commit()
        self.session.close()

    def _populate_instrument_table(self, ticker_attributes_dict: dict) -> None:
        """
        Adds rows to the instruments table in the database. Assumes that the given dictionary has tickers as keys and
        attribute sub-dictionaries as values
        :param ticker_attributes_dict: dict {ticker, {attribute_name: attribute_value}, ...}
        :return: None
        """
        self.session.add_all(
            [Instrument(ticker=ticker, **ticker_attributes_dict[ticker])
             for ticker in ticker_attributes_dict.keys()]
        )
        self.session.commit()
        self.session.close()

    def adjusting_instrument_attributes(self, ticker_attributes_dict: dict) -> None:
        """
        Changes attributes for instruments that already exists in the database. Assumes that the given dictionary has
        tickers (with capital letters) as keys and attribute sub-dictionaries as values
        :param ticker_attributes_dict: dict {ticker, {attribute_name: attribute_value}, ...}
        :return: None
        """
        # query all instrument rows that will have their attributes changed
        q_instruments = self.session.query(Instrument) \
            .filter(
            Instrument.ticker.in_(ticker_attributes_dict.keys())) \
            .all()

        for instrument in q_instruments:
            new_attributes = ticker_attributes_dict[instrument.ticker]  # {attribute_name: attribute_value}
            for attribute in new_attributes.keys():
                if attribute in Instrument.__table__.c.keys():
                    setattr(instrument, attribute, new_attributes[attribute])
                else:
                    raise ValueError("'{}' does not exist as an attribute of Instrument class".format(attribute))
        self.session.commit()
        self.session.close()

    def _populate_data_table(self, df: pd.DataFrame = None,
                             data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}=None,
                             table_df_dict: dict = None, replace: bool = False) -> None:
        """
        Inserts rows in the data table class:
        1) records the date of the refresh in the instrument table
        2) replaces the data in the database with the data in the given DataFrame or first removes the overlaps before
        insert
        3) formats the DataFrame to be in line with the data table and insert it to the database
        :param df: DataFrame
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :param table_df_dict: {DataFrame: data table}
        :param replace: bool
        :return: None
        """

        if table_df_dict is not None:
            if com.count_not_none(df, data_table) > 0:
                raise ValueError("when 'table_df_dict' is specified, please don't specify 'df' or 'table'")
            for data_table, df in table_df_dict.items():
                # recall the function recursively
                self._populate_data_table(df=df, data_table=data_table)
        else:
            if com.count_not_none(df, data_table) != 2:
                raise ValueError("when 'table_df_dict' is not specified, please specify 'df' and 'table'")
            logger.info("Add rows to '{}' table".format(data_table.__tablename__))
            # remove all the overlapping dates
            if replace:  # removes the overlaps from the database
                # remove the dates from the database that exists in the DataFrame for the relevant tickers
                self._delete_rows_in_data_table_overlapping_df(df=df, data_table=data_table)
            else:  # removes the overlaps from the input
                # set values to be removed from the DataFrame to nan
                ticker_last_date_dict = self.get_latest_as_of_date_dict(tickers=list(df), table=data_table)
                # column-wise, for each date before the available date in the database, set the values to nan (to be
                # removed when converting the DataFrame to an SQL table)
                df = nan_before_dates(df=df, col_name_date_dict=ticker_last_date_dict)

            # format the DataFrame to be in line with the SQL data table
            df_sql = self._reformat_df_to_sql_table(df=df, data_table=data_table)

            # insert the new rows into the data table
            df_sql.to_sql(data_table.__tablename__, self.session.get_bind(), if_exists='append', index=False)

            # update the refresh columns for the tickers in the DataFrame
            self._record_refresh(tickers=list(df), table=data_table)
            self.session.commit()
            self.session.close()

    def _record_refresh(self, tickers: list, table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}) -> None:
        """
        In the instrument table, set the latest refresh date for the relevant table to today()
        :param tickers: list of str
        :param table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: None
        """
        logger.debug("Record refresh for {} ticker(s) for '{}' table".format(len(tickers), table.__tablename__))
        self.adjusting_instrument_attributes(ticker_attributes_dict=
                                             {ticker: {table.refresh_info_column_name: date.today()}
                                              for ticker in tickers}
                                             )

    def _reformat_df_to_sql_table(self, df: pd.DataFrame, data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}) -> pd.DataFrame:
        """
        Reformat the given DataFrame (assumed to have tickers as column names and dates as index) to be in line with the
        given data table class
        :param df: DataFrame
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: DataFrame
        """
        logger.debug("Reformat the loaded DataFrame to be in line with a SQL table")
        # 'melt' the DataFrame i.e. a 'reverse pivot'
        df_index_reset = df.reset_index()
        df_index_reset[list(df_index_reset)[0]] = df_index_reset[list(df_index_reset)[0]].dt.date  # datetime to date
        sql_table_df = pd.melt(df_index_reset, id_vars=[list(df_index_reset)[0]])
        first_col_name = list(sql_table_df)[0]

        # drop the nans in the 'value' column
        sql_table_df.dropna(subset=['value'], inplace=True)

        # add data source and created_at column
        sql_table_df['source'] = self.source
        sql_table_df['created_at'] = date.today()

        # replace the ticker with the corresponding instrument id
        ticker_id_dict = self.get_ticker_instrument_attribute_dict(tickers=list(set(sql_table_df['variable'].values)),
                                                                   attribute='id')
        sql_table_df.replace({'variable': ticker_id_dict}, inplace=True)

        # reshuffle columns and set column names
        sql_table_df = sql_table_df[[first_col_name, 'value', 'source', 'created_at', 'variable']]
        sql_table_df.columns = data_table.__table__.columns.keys()[1:]  # first key is 'id' and should not be provided
        return sql_table_df

    @property
    def source(self):
        return self._source

    @property
    def as_of_date(self):
        return self._as_of_date

    @as_of_date.setter
    def as_of_date(self, as_of_date: date):
        if as_of_date is None:
            self._as_of_date = (date.today() - BDay(1)).date() if date.today().weekday() in [5, 6] else date.today()
        elif as_of_date <= date.today():
            self._as_of_date = as_of_date
        else:
            raise ValueError("as_of_date needs to be today or earlier")


class YahooFinanceFeeder(_DatabaseFeeder):
    _new_old_key_map = {'instrument_type': 'quoteType', 'short_name': 'shortName', 'long_name': 'longName',
                        'address': 'address1', 'description': 'longBusinessSummary'}
    _old_new_value_per_old_key_map = {'quoteType': {'EQUITY': 'STOCK', 'CURRENCY': 'FX'}}
    _data_tables = [OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, Split]

    def __init__(self, **kwargs):
        super().__init__(source='YAHOO', **kwargs)
        self._yf_con = _YahooFinanceConnection()

    def _download_instrument_info_dict(self, tickers: list):
        return self._yf_con.get_info_dict(tickers=tickers)

    def _download_data_dict(self, tickers: list, start_date: {date, None}, end_date: {date, None}):
        name_data_table_map = {'Open': OpenPrice, 'High': HighPrice, 'Low': LowPrice, 'Close': ClosePrice,
                               'Volume': Volume, 'Dividends': Dividend, 'Stock Splits': Split}
        ohlc_volume = self._yf_con.get_ohlc_volume(tickers=tickers, start_date=start_date, end_date=end_date)
        dividend_split = self._yf_con.get_dividend_and_splits(tickers=tickers, start_date=start_date, end_date=end_date)
        return {v: ohlc_volume.get(k, dividend_split.get(k)) for k, v in name_data_table_map.items()}


def make_ticker_list(ticker: {str, list})->list:
    """
    Returns a list of tickers with capitalized letters
    :param ticker:
    :return: list
    """
    # convert if necessary to list
    ticker = ticker if isinstance(ticker, list) else [ticker]
    ticker = [ticker.upper() for ticker in ticker.copy()]  # capital letters
    if len(set(ticker)) < len(ticker):
        raise ValueError("the given tickers contains duplicates")
    else:
        return ticker














