"""
financial_database_manager.py
"""
import pandas as pd
import pandas.core.common as com
from pandas.tseries.offsets import BDay
from datetime import date, timedelta

from sqlalchemy import func, or_, and_

# my data base configurations
from database2.base import session_factory
from database2.models import Instrument, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend, \
    data_table_instrument_types, available_data_sources
from database2.yahoo_finance import YahooFinanceConnection as _YahooFinanceConnection

# other tools
from tools.general_tools import user_picks_element_from_list, list_grouper, reverse_dict
from tools.dataframe_tools import nan_before_dates



class FinancialDatabaseManager:

    def __init__(self):
        self.session = session_factory()

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
            self.session.delete(instrument)
        self.session.commit()
        self.session.close()

    def delete_rows_in_data_table(self, df: pd.DataFrame,
                                  data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}):
        """
        Deletes the rows in the given data table class corresponding to overlaps in the given DataFrame
        :param df: DataFrame (columns = instrument tickers, index = DatetimeIndex)
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: None
        """

        # create a list of all the instrument ids corresponding to the tickers
        instrument_ids = [inst_id for inst_id in
                          self.get_ticker_instrument_attribute_dict(tickers=list(df), attribute='id').values()]

        # delete all data table rows for each relevant tickers with a date between the DataFrame index
        start_date = min(df.index)
        end_date = max(df.index)

        for sub_instrument_id_list in list_grouper(instrument_ids, 500):
            self.session.query(data_table) \
                .filter(
                and_(
                    data_table.instrument_id.in_(sub_instrument_id_list),
                    and_(data_table.date >= start_date, data_table.date <= end_date)
                )
            ).delete(synchronize_session=False)
        self.session.commit()
        self.session.close()

    def ticker_existence(self, tickers: {str, list}, data_source: str) -> tuple:
        """
        Returns a tuple with three lists: 1) list of existing tickers 2) list of missing tickers 3) list of existing tickers
        but with a different data source than the one provided
        :param tickers: str or list
        :param data_source: str
        :return: (list, list, list)
        """

        tickers = make_ticker_list(tickers)

        # filter out the tickers and associated data sources
        q_existing_tickers = self.session.query(
            Instrument.ticker,
            Instrument.data_source) \
            .filter(
            Instrument.ticker.in_(tickers)
        ).all()
        self.session.close()

        # create the result
        existing_tickers = [q_tup[0] for q_tup in q_existing_tickers]
        missing_tickers = list(set(tickers).difference(existing_tickers))
        existing_tickers_diff_source = [q_tup[0] for q_tup in q_existing_tickers if q_tup[1] != data_source.upper()]
        return existing_tickers, missing_tickers, existing_tickers_diff_source

    def find_unrefreshed_tickers(self, tickers: {str, list},
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
                unrefreshed_tickers.extend(self.find_unrefreshed_tickers(tickers=tickers, data_table=d_tab))
            return list(set(unrefreshed_tickers))
        else:
            tickers = make_ticker_list(tickers)  # TODO convert somewhere else?

            eligible_instrument_types = data_table_instrument_types[data_table]  # list of eligible instrument types
            q_ticker = self.session.query(Instrument.ticker)  # initial query for the tickers

            # latest weekday
            bd_today = date.today() - BDay(1) if date.today().weekday() in [5, 6] else date.today()

            for tickers_sub_list in list_grouper(tickers, 500):
                sub_q_ticker = q_ticker \
                    .filter(
                    and_(
                        Instrument.ticker.in_(tickers_sub_list),  # tickers should be in the sub list
                        Instrument.instrument_type.in_(eligible_instrument_types),  # the instrument type is eligible
                        or_(
                            getattr(Instrument, data_table.refresh_info_column_name) == None,
                            # the ticker has no data in the table
                            getattr(Instrument, data_table.refresh_info_column_name) < bd_today
                            # the latest refresh was in the past
                        )
                    )
                ).all()
                unrefreshed_tickers.extend([q_tick[0] for q_tick in sub_q_ticker])
            self.session.close()
            return unrefreshed_tickers

    def get_ticker_instrument_attribute_dict(self, tickers: {str, list}, attribute: str) -> dict:
        """
        Assumes that ticker is a string or a list of strings and attribute is a string (e.g. Instrument.sector will
        return a dictionary like {'ticker 1': 'sector A', 'ticker 2': 'sector B' ,...}).
        Returns a dictionary with tickers as keys and the specific attribute as values
        :param tickers:
        :param attribute: str
        :return: dict
        """

        tickers = make_ticker_list(tickers)
        ticker_attribute_dict = {}  # initializing the dictionary

        # to make the requests smaller, we need to split the ticker list into sub list
        for ticker_sub_list in list_grouper(tickers, 500):
            query_ticker_attribute = self.session.query(Instrument.ticker, getattr(Instrument, attribute)) \
                .filter(
                Instrument.ticker.in_(ticker_sub_list)) \
                .order_by(Instrument.ticker)
            ticker_attribute_dict.update(dict(query_ticker_attribute))
        self.session.close()
        return ticker_attribute_dict

    def get_latest_date_dict(self, tickers: {str, list},
                             table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}) -> dict:
        """
        Returns a dictionary with tickers as keys and the latest date with a value in the given data table as values
        :param tickers: str, list
        :param table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}
        :return: dict
        """

        tickers = make_ticker_list(tickers)
        ticker_id_dict = self.get_ticker_instrument_attribute_dict(tickers=tickers, attribute='id')

        # make the initial query for instrument id and maximum of the dates to later be filtered and grouped by
        q_latest_date = {}
        q_instrument_id_max_date = self.session.query(
            table.instrument_id,
            func.max(table.date)
        )
        for instrument_id_sub_list in list_grouper(ticker_id_dict.values(), 500):
            # first query the latest observation date for the close price
            sub_q_latest_close_date = q_instrument_id_max_date \
                .filter(
                table.instrument_id.in_(instrument_id_sub_list)
            ).group_by(
                table.instrument_id
            )
            q_latest_date.update(sub_q_latest_close_date)
        self.session.close()

        # instead of returning {instrument.id: latest date} return {instrument.ticker: latest date}
        id_ticker_dict = reverse_dict(dict_=ticker_id_dict)  # {ticker: id} -> {id: ticker}
        return {id_ticker_dict[_id]: latest_date for (_id, latest_date) in q_latest_date.items()}

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

            tickers = make_ticker_list(ticker=tickers)  # TODO convert somewhere else?

            latest_dates = self.get_latest_date_dict(tickers=tickers, table=data_table).values()
            if len(latest_dates) < len(tickers):  # meaning at least one ticker does not have a row in the data table
                return None
            else:
                return min(latest_dates)


class _DatabaseFeeder(FinancialDatabaseManager):

    def __init__(self, source: str = None):
        super().__init__()
        # set the data source (can only be set once and is read only)
        if source is None:
            self._source = source
        elif source.upper() in available_data_sources:
            self._source = source.upper()
        else:
            raise ValueError("'source' needs to be one of '%s'" % "', '".join(available_data_sources))
        self._new_old_key_translator = {}
        self._old_new_value_translator_per_key = {}
        self._data_tables = None

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

        adj_info = {}
        for ticker in info_dict.keys():
            sub_adj_info = {}
            for new_key in self._new_old_key_translator.keys():
                old_key = self._new_old_key_translator[new_key]
                old_value = info_dict[ticker].get(old_key, None)
                new_value = self._old_new_value_translator_per_key[old_key].get(old_value, old_value) \
                    if old_key in self._old_new_value_translator_per_key.keys() \
                    else old_value
                sub_adj_info.update({new_key: new_value})
            adj_info[ticker] = sub_adj_info
        return adj_info

    def add_instrument(self, tickers) -> None:

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
                pass
            elif chosen_task == tasks[2]:  # Refresh
                # load the daily data (based on the latest date in the database) and remove all loaded daily data that
                # overlaps with the database

                pass
            elif chosen_task == tasks[3]:
                # do nothing
                pass
            else:
                raise ValueError(f"'{chosen_task}' is not a recognized task")
            print(chosen_task)

    def refresh_data(self, tickers: {str, list}, as_of_date: date = None, replace: bool = False):

        as_of_date = date.today() - timedelta(days=1) if as_of_date is None else as_of_date
        if as_of_date > date.today():
            raise ValueError("Can't refresh as of a date that is in the future")

        # get the correct start date
        start_date = self.get_eligible_start_date(tickers=tickers, data_table=self._data_tables)




    def populate_instrument_table(self, ticker_attributes_dict: dict) -> None:
        """
        Adds rows to the instruments table in the database. Assumes that the given dictionary has tickers as keys and
        attribute sub-dictionaries as values
        :param ticker_attributes_dict: dict {ticker, {attribute_name: attribute_value}, ...}
        :return: None
        """
        instrument_list = [Instrument(ticker=ticker, **ticker_attributes_dict[ticker]) for ticker in
                           ticker_attributes_dict.keys()]
        self.session.add_all(instrument_list)
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
                setattr(instrument, attribute, new_attributes[attribute])
        self.session.commit()
        self.session.close()

    def populate_data_table(self, df: pd.DataFrame = None,
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
            for df, data_table in table_df_dict.items():
                # recall the function recursively
                self.populate_data_table(df=df, data_table=data_table)
        else:
            if com.count_not_none(df, data_table) != 2:
                raise ValueError("when 'table_df_dict' is not specified, please specify 'df' and 'table'")

            # update the refresh columns for the tickers in the DataFrame
            self.record_refresh(tickers=list(df), table=data_table)

            # remove all the overlapping dates
            if replace:  # removes the overlaps from the database
                # remove the dates from the database that exists in the DataFrame for the relevant tickers
                self.delete_rows_in_data_table(df=df, data_table=data_table)
            else:  # removes the overlaps from the input
                # set values to be removed from the DataFrame to nan
                ticker_last_date_dict = self.get_latest_date_dict(tickers=list(df), table=data_table)
                # column-wise, for each date before the available date in the database, set the values to nan (to be removed
                # when converting the DataFrame to an SQL table)
                df = nan_before_dates(df=df, col_name_date_dict=ticker_last_date_dict)

            # format the DataFrame to be in line with the SQL data table
            df_sql = self.reformat_df_to_sql_table(df=df, data_table=data_table)

            # insert the new rows into the data table
            df_sql.to_sql(data_table.__tablename__, self.session.get_bind(), if_exists='append', index=False)
            self.session.commit()
            self.session.close()

    def record_refresh(self, tickers: list, table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend}) -> None:
        """
        In the instrument table, set the latest refresh date for the relevant table to today()
        :param tickers: list of str
        :param table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :return: None
        """
        # TODO have a variable called 'as_of_date' by default set to yesterday (this is to avoid loading data before
        #  close)
        ticker_refresh_info_dict = {ticker: {table.refresh_info_column_name: date.today()} for ticker in tickers}
        self.adjusting_instrument_attributes(ticker_attributes_dict=ticker_refresh_info_dict)

    def reformat_df_to_sql_table(self, df: pd.DataFrame, data_table: {OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend},
                                 comment: str = None) -> pd.DataFrame:
        """
        Reformat the given DataFrame (assumed to have tickers as column names and dates as index) to be in line with the
        given data table class
        :param df: DataFrame
        :param data_table: OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
        :param comment: str
        :return: DataFrame
        """
        # 'melt' the DataFrame i.e. a 'reverse pivot'
        sql_table_df = pd.melt(df.reset_index(), id_vars=[list(df.reset_index())[0]])

        # drop the nans in the 'value' column
        sql_table_df.dropna(subset=['value'], inplace=True)

        # add data source and comment column
        sql_table_df['data_source'] = self.source
        comment = 'Inserted {}'.format(date.today()) if comment is None else comment
        sql_table_df['comment'] = comment

        # replace the ticker with the corresponding instrument id
        ticker_id_dict = self.get_ticker_instrument_attribute_dict(tickers=list(set(sql_table_df['variable'].values)),
                                                                   attribute='id')
        sql_table_df.replace({'variable': ticker_id_dict}, inplace=True)

        # reshuffle columns and set column names
        sql_table_df = sql_table_df[['index', 'value', 'data_source', 'comment', 'variable']]
        sql_table_df.columns = data_table.__table__.columns.keys()[1:]  # first key is 'id' and should not be provided
        return sql_table_df

    @property
    def source(self):
        return self._source


class YahooFinanceFeeder(_DatabaseFeeder):

    def __init__(self):
        super().__init__(source='YAHOO')
        self._yf_con = _YahooFinanceConnection()

    def _download_instrument_info_dict(self, tickers: list):
        yf_info_dict = self._yf_con.get_info_dict(tickers=tickers)
        # fetch the relevant keys and rename some to be in line with the database

    def _yf_info_dict_mapper(self, yf_info_dict: dict):
        pass

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


yf = YahooFinanceFeeder()



# populate_instrument_table(
#     {'gs': {'data_source': source, 'instrument_type': 'stock', 'asset_class': 'equity', 'currency': 'usd', 'short_name': 'Goldman Sachs'},
#      'hm-b.st': {'data_source': source, 'instrument_type': 'stock', 'asset_class': 'equity', 'currency': 'SEK', 'short_name': 'H&M', 'sector': 'clothes and stuff'}}
# )
# fx = Instrument('USD/EUR', 'yahoo', instrument_type='fx', asset_class='fx', currency='usd', short_name='USD per EUR')
# session.add(fx)
#
# gs_instrument = session.query(Instrument).filter(Instrument.ticker == 'GS').first()
# gs_instrument.close_prices = [ClosePrice(date(1992, 1, 1), 45, 'BLOOMBERG'), ClosePrice(date(1993, 1, 1), 54, 'BLOOMBERG')]
# gs_instrument.dividends = [Dividend(date(2000, 1, 1), 4, 'BLOOMBERG'), Dividend(date(2003, 1, 1), 1, 'BLOOMBERG')]
#
# hm_instrument = session.query(Instrument).filter(Instrument.ticker == 'HM-B.ST').first()
# hm_instrument.close_prices = [ClosePrice(date(1980, 1, 1), 2332, 'BLOOMBERG'), ClosePrice(date(1970, 1, 1), 500, 'BLOOMBERG')]
# hm_instrument.dividends = [Dividend(date(2010, 1, 1), 1, 'BLOOMBERG')]
#


# adjusting_instrument_attributes({'GS': {Dividend.refresh_info_column_name: None}})
# adjusting_instrument_attributes({'HM-B.ST': {Dividend.refresh_info_column_name: None}})
#
#
# __tickers = ['gs', 'hm-b.st']
# eligible_tickers = find_unrefreshed_tickers(tickers=__tickers, data_table=[Dividend, ClosePrice])
# print(eligible_tickers)
# print(get_eligible_start_date(tickers=eligible_tickers, data_table=[Dividend, ClosePrice]))
#
# print(Dividend.__table__.columns.keys())
#
# # print all instruments
# instruments = session.query(Instrument).all()
# for inst in instruments:
#     print(f'{inst.ticker}({inst.data_source}) is in the {inst.sector} sector and {inst.industry} industry')
#
#
# import numpy as np
# data = pd.DataFrame({'GS': [400, 400], 'HM-B.ST': [300, 400]}, index=[date(2001, 9, 11), date(2001, 9, 12)])
#
# populate_data_table(df=data, data_table=Dividend)
#
# print(find_unrefreshed_tickers(tickers=__tickers, data_table=Dividend))
#
# session.commit()
# session.close()











