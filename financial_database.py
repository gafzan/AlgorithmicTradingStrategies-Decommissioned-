from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
import logging
import yfinance
from datetime import date, datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np

# my own modules
from models_db import Base, Underlying, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
from general_tools import capital_letter_no_blanks, list_grouper, extend_dict, reverse_dict, progression_bar
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class _StaticFinancialDatabase:
    """Class definition of _StaticFinancialDatabase
    No functionality for creating, reading or updating data. However, you can reset the database thus deleting
    all its content."""

    def __init__(self, database_name: str, database_echo=False):
        self._database_name = database_name
        self._database_echo = database_echo
        engine = create_engine(self.database_name, echo=self.database_echo)
        Base.metadata.create_all(engine)  # create the database tables (these are empty at this stage)
        Session = sessionmaker(bind=engine)  # ORM's 'handle' to the database (bound to the engine object)
        self._session = Session()  # whenever you need to communicate with the database you instantiate a Session

    @staticmethod
    def handle_ticker_input(ticker: {str, list}, convert_to_list=False, sort=False) -> {str, list}:
        """Assumes that ticker is either a string or a list and convert_to_list is bool. Returns a string or a list
        of strings where all the strings have capital letters and blanks have been replaced with '_'."""
        ticker = capital_letter_no_blanks(ticker)
        if isinstance(ticker, list) and sort:
            ticker.sort()
        if isinstance(ticker, str) and convert_to_list:
            ticker = [ticker]
        return ticker

    # ------------------------------------------------------------------------------------------------------------------
    # get set functionality

    # session is read-only
    @property
    def session(self):
        return self._session

    @property
    def database_name(self):
        return self._database_name

    @property
    def database_echo(self):
        return self._database_echo

    # when either database_name or database_echo changes, the session attribute resets using the _set_session method
    @database_name.setter
    def database_name(self, database_name: str):
        self._database = database_name
        self._set_session()

    @database_echo.setter
    def database_echo(self, database_echo: bool):
        self._database_echo = database_echo
        self._set_session()

    def _set_session(self) -> None:
        engine = create_engine(self.database_name, self.database_echo)
        Base.metadata.create_all(engine)  # Create the database tables (these are empty at this stage)
        Session = sessionmaker(bind=engine)  # ORM's 'handle' to the database (bound to the engine object)
        self._session = Session()

    def resetting_database(self) -> None:
        """Resets the database, meaning that contents in all tables will be deleted."""
        logger.info('Resetting database ({}).'.format(self.database_name))
        Base.metadata.drop_all(self.session.get_bind())
        Base.metadata.create_all(self.session.get_bind())
        return

    def __repr__(self):
        return f"<_StaticFinancialDatabase(name = {self.database_name})>"


class FinancialDatabase(_StaticFinancialDatabase):
    """Class definition for FinancialDatabase.
    This class allows for reading and deleting data but not for creating or updating data."""

    # ------------------------------------------------------------------------------------------------------------------
    # methods for Underlying table

    def underlying_exist(self, ticker: str) -> bool:
        """Assumes that ticker is a string. Returns True if there is an Underlying row that is represented by the
        ticker, else returns False."""
        ticker = self.handle_ticker_input(ticker)
        logger.debug("Checking for existence of the ticker '{}'.".format(ticker))
        return self.session.query(Underlying).filter(Underlying.ticker == ticker).count() > 0

    def delete_underlying(self, tickers: {str, list}) -> None:
        """Assumes that tickers is either a string or a list of strings. Deletes all Underlying rows corresponding to
        the ticker(s). This will also remove all rows from tables that are subclasses to Underlying."""
        tickers = self.handle_ticker_input(tickers, convert_to_list=True)
        logger.info("Deletes {} ticker(s) from the database.\nTickers: {}".format(len(tickers), tickers))
        for ticker in tickers:
            if self.underlying_exist(ticker):
                query_underlying = self.session.query(Underlying).filter(Underlying.ticker == ticker).first()
                logger.debug('Delete {} from the database.'.format(ticker))
                self.session.delete(query_underlying)
            else:
                logger.info('{} does not exist in the database.'.format(ticker))
        self.session.commit()
        return

    def _delete_open_high_low_close_volume_dividend_data(self, ticker_list: list, start_date: {datetime, date},
                                                         end_date: {datetime, date}) -> None:
        """Assumes that tickers is a list of strings and start_date and end_date are of type datetime. Deletes all rows
        in OpenPrice, HighPrice, LowPrice, ClosePrice, Volume and Dividend table for the given tickers from start_date
        to end_date."""
        logger.debug("Deleting data for {} ticker(s) between {} and {}.\nTickers: {}".format(len(ticker_list),
                                                                                             str(start_date)[:10],
                                                                                             str(end_date)[:10],
                                                                                             ticker_list))
        table_list = [OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend]
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        underlying_id_list = ticker_underlying_id_dict.values()
        for table in table_list:
            if table == Dividend:  # in case of Dividend table use 'ex-dividend date'
                date_between_start_and_end = table.ex_div_date.between(start_date, end_date)
            else:  # else use observation date
                date_between_start_and_end = table.obs_date.between(start_date, end_date)
            # for each underlying id, query the data that exists between the start and end date and remove the content
            for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
                self._session.query(table)\
                    .filter(
                    and_(
                        table.underlying_id.in_(underlying_id_sub_list),
                        date_between_start_and_end)
                )\
                    .delete(synchronize_session=False)
        self.session.commit()
        self._update_obs_date_and_dividend_history_status(ticker_list)

    def get_ticker(self, underlying_attribute_dict: dict = None) -> list:
        """Assume that underlying_attribute_dict is a dictionary with Underlying.attribute_name (e.g. Underlying.sector)
        as ‘key’ and attribute value (e.g. ‘INDUSTRIALS’) as ‘value’. Returns a list of tickers (each a string) who
        shares the attributes.
        To find the union of tickers with two values of attributes (e.g. all tickers in ‘INDUSTRIALS’ and ‘ENERGY’)
        simply have the 'key' (Underlying.sector) point to a list ([‘INDUSTRIALS’, ‘ENERGY’]).
        The resulting dictionary will hence look like {Underlying.sector:  [‘INDUSTRIALS’, ‘ENERGY’]}.
        Adding several keys will mean that you are taking the intersection between the attributes. E.g. inputting the
        dictionary {Underlying.sector:  [‘INDUSTRIALS’, ‘ENERGY’], Underlying.currency: ‘JPY’} will lead to the method
        returning tickers for stocks in the Industrial and Energy sector that are all denominated in JPY."""
        if underlying_attribute_dict is None:
            underlying_attribute_dict = {}  # selection will be based on no attributes i.e. select all tickers
        underlying_attribute_list = underlying_attribute_dict.keys()

        logger_message = 'Filtering tickers based on:'
        for dict_key in underlying_attribute_list:
            logger_message += "\n{} = {}".format(dict_key, underlying_attribute_dict[dict_key])
        logger.info(logger_message)

        # TODO need to split the request for large number of tickers
        query_ticker = self.session.query(Underlying.ticker)
        for underlying_attribute in underlying_attribute_list:
            underlying_attribute_value = underlying_attribute_dict[underlying_attribute]  # e.g. 'ENERGY' if sector
            if not isinstance(underlying_attribute_value, list):
                underlying_attribute_value = [underlying_attribute_value]  # string -> [string]
            underlying_attribute_value = capital_letter_no_blanks(underlying_attribute_value)
            query_ticker = query_ticker.filter(underlying_attribute.in_(underlying_attribute_value))
        query_ticker.order_by(Underlying.ticker)
        ticker_list = [tup[0] for tup in query_ticker]  # extract the ticker string from the result
        logger.info("{} ticker(s) selected.".format(len(ticker_list)))
        return ticker_list

    def get_ticker_underlying_attribute_dict(self, tickers: {str, list}, underlying_attribute) -> dict:
        """Assumes that ticker is a string or a list of strings and attribute is of type
        sqlalchemy.orm.attributes.InstrumentedAttribute (e.g. Underlying.sector will return a dictionary like
        {'ticker 1': 'sector A', 'ticker 2': 'sector B' ,...}).
        Returns a dictionary with tickers as keys and the specific attribute as values."""
        tickers = self.handle_ticker_input(tickers, convert_to_list=True)
        ticker_attribute_dict = {}  # initializing the dictionary

        # to make the requests smaller, we need to split the ticker list into sub list
        for ticker_sub_list in list_grouper(tickers, 500):
            query_ticker_attribute = self.session.query(Underlying.ticker, underlying_attribute) \
                .filter(
                Underlying.ticker.in_(ticker_sub_list))\
                .order_by(Underlying.ticker)
            ticker_attribute_dict = extend_dict(ticker_attribute_dict, dict(query_ticker_attribute))
        return ticker_attribute_dict

    def _update_obs_date_and_dividend_history_status(self,  tickers: list) -> None:
        """Assumes tickers is a list of strings. For each ticker, method assign to the Underlying table 1) latest
        observation date, 2) latest observation date with value, 3) oldest observation date and 4) first ex-dividend
        date (if any)."""
        logger.debug('Updating oldest and latest observation date and first ex-dividend date for {} ticker(s).'
                     '\nTicker: {}'.format(len(tickers), tickers))
        underlying_id_list = self.get_ticker_underlying_attribute_dict(tickers, Underlying.id).values()
        ticker_counter = 0  # only used for the logger
        for underlying_id in underlying_id_list:
            query_underlying = self.session.query(Underlying).filter(Underlying.id == underlying_id).first()
            # check dividend history
            if not query_underlying.has_dividend_history:
                # see if there are any newly recorded ex-dividend dates.
                query_dividend_amount_ex_dividend_date = self.session\
                    .query(Dividend.dividend_amount,
                           Dividend.ex_div_date)\
                    .filter(Dividend.underlying_id == underlying_id)
                if query_dividend_amount_ex_dividend_date.count() > 0:
                    logger.debug("Setting dividend history status for {} to True.".format(tickers[ticker_counter]))
                    query_underlying.has_dividend_history = True

                    # find and record the first ex-dividend date
                    first_ex_div_date = query_dividend_amount_ex_dividend_date.order_by(Dividend.ex_div_date).first()[1]
                    logger.debug("First ex-dividend date for {} was {}.".format(tickers[ticker_counter],
                                                                                str(first_ex_div_date)[:10]))
                    query_underlying.first_ex_div_date = date(year=first_ex_div_date.year,
                                                              month=first_ex_div_date.month,
                                                              day=first_ex_div_date.day)
            # check observation dates
            query_obs_date = self.session.query(ClosePrice.obs_date).filter(ClosePrice.underlying_id == underlying_id)
            query_obs_date_with_value = query_obs_date.filter(ClosePrice.close_quote.isnot(None))
            latest_obs_date = query_obs_date.order_by(ClosePrice.obs_date.desc()).first()[0]
            latest_obs_date_with_values = query_obs_date_with_value.order_by(ClosePrice.obs_date.desc()).first()[0]
            logger.debug('Setting latest observation date for {} to {}.'.format(tickers[ticker_counter],
                                                                                str(latest_obs_date)[:10]))
            query_underlying.latest_observation_date = latest_obs_date
            logger.debug('Setting latest observation date (with value) for {} to {}.'.format(tickers[ticker_counter],
                                                                                             str(latest_obs_date)[:10]))
            query_underlying.latest_observation_date_with_values = latest_obs_date_with_values
            if query_underlying.oldest_observation_date is None:
                oldest_obs_date = query_obs_date.order_by(ClosePrice.obs_date).first()[0]
                logger.debug('Setting oldest observation date (with value) for {} to {}.'.format(
                    tickers[ticker_counter], str(oldest_obs_date)[:10]))
                query_underlying.oldest_observation_date = oldest_obs_date
            ticker_counter += 1
        self.session.commit()
        return

    # ------------------------------------------------------------------------------------------------------------------
    # methods for OpenPrice, HighPrice, LowPrice, ClosePrice, Volume and Dividend tables

    def _input_check(self, tickers: {str, list}, start_date: {date, datetime}, end_date: {date, datetime}) -> tuple:
        """This method checks some of the inputs for _get_open_high_low_close_volume_dividend_df method. Returns a
        tuple with the inputs that have been adjusted if applicable."""
        # adjust inputs
        tickers = self.handle_ticker_input(tickers, convert_to_list=True, sort=False)
        if len(tickers) == 0:
            raise TypeError('The ticker list was empty.')
        tickers_not_in_database = []
        for ticker in tickers:
            if not self.underlying_exist(ticker):
                tickers_not_in_database.append(ticker)
        if len(tickers_not_in_database) > 0:
            raise ValueError(
                "{} ticker(s) are missing from the database.\nTickers: {}".format(len(tickers_not_in_database),
                                                                                  tickers_not_in_database))
        if start_date is None:
            # Pick the oldest observation date available
            start_date = min(
                self.get_ticker_underlying_attribute_dict(tickers, Underlying.oldest_observation_date).values())
        if end_date is None:
            # Pick the latest observation date with data available
            end_date = max(self.get_ticker_underlying_attribute_dict(tickers,
                                                                     Underlying.latest_observation_date_with_values).values())
        return tickers, start_date, end_date

    def _get_open_high_low_close_volume_dividend_df(self, table, tickers: {str, list}, start_date: {date, datetime},
                                                    end_date: {date, datetime}, currency: str)->pd.DataFrame:
        tickers, start_date, end_date = self._input_check(tickers, start_date, end_date)
        logger.debug('Get {} data for {} ticker(s)'.format(table.__tablename__, len(tickers))
                     + logger_time_interval_message(start_date, end_date))

        # need to add an extra day otherwise the 'between' function below does not capture the end date
        end_date = end_date + timedelta(1)

        # dictionary that holds the name of the value column
        value_column_name_dict = {OpenPrice: 'open_quote', HighPrice: 'high_quote', LowPrice: 'low_quote',
                                  ClosePrice: 'close_quote', Volume: 'volume_quote', Dividend: 'dividend_amount'}

        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(tickers, Underlying.id)
        underlying_id_list = ticker_underlying_id_dict.values()
        result_df = None
        # for each underlying id, query data from the requested table
        for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
            if table == Dividend:
                table_date = table.ex_div_date
            else:
                table_date = table.obs_date
            # query date, value and underlying.id for the specific tickers and between the specific start and end date
            query_date_value_id = self.session.query(table_date,
                                                     value_column_name_dict[table],
                                                     table.underlying_id)\
                .filter(
                and_(table.underlying_id.in_(underlying_id_sub_list),
                     table_date.between(start_date, end_date))
            )
            sub_df = pd.read_sql_query(query_date_value_id.statement, self.session.get_bind())
            if result_df is None:  # first sub list, no need to concatenate the DataFrame
                result_df = sub_df
            else:
                result_df = pd.concat([result_df, sub_df], ignore_index=True)

        # pivot the DataFrame s.t. dates are the index and Underlying.id are the column headers
        result_pivoted_df = result_df.pivot(index=result_df.columns[0], columns=result_df.columns[2],
                                            values=value_column_name_dict[table])
        # change the column names from Underlying.id to ticker
        underlying_id_ticker_dict = reverse_dict(ticker_underlying_id_dict)
        column_names = [underlying_id_ticker_dict[underlying_id] for underlying_id in
                        result_pivoted_df.columns]
        result_pivoted_df.columns = column_names

        # handle tickers that has no data
        result_pivoted_df = self._handle_missing_data(tickers, result_pivoted_df)

        # if all columns are nan, remove the entire row
        result_pivoted_df.dropna(inplace=True, how='all')

        # currency convert if applicable
        if isinstance(currency, str) and table != Volume:
            return self._currency_convert_df(result_pivoted_df, currency)
        else:
            return result_pivoted_df

    @staticmethod
    def _handle_missing_data(original_ticker_list: list, values_df: pd.DataFrame) -> pd.DataFrame:
        """Assume that original_ticker_list is a list of tickers that needs to be column names in values_df. If a ticker
        does not exist as a column name, insert a column with NaN as values. Returns a DataFrame."""
        missing_ticker_list = list(set(original_ticker_list).difference(list(values_df)))
        for missing_ticker in missing_ticker_list:
            values_df.insert(0, missing_ticker, np.nan)  # NaN for each date
        values_df = values_df[original_ticker_list]  # rearrange the column names
        return values_df

    def _currency_convert_df(self, values_df: pd.DataFrame, currency: str) -> pd.DataFrame:
        """Assumes that values_df is a DataFrame with tickers as column headers and dates as index, start_date and
        end_date is either of type date or datetime and currency is a string. First, the method finds the correct FX
        data based on the currency that each ticker is quoted in. The method then converts the values in the DataFrame.
        """
        price_currency = capital_letter_no_blanks(currency)
        logger.debug('Converts DataFrame to {}.'.format(price_currency))
        ticker_currency_dict = self.get_ticker_underlying_attribute_dict(list(values_df), Underlying.currency)
        ticker_fx_ticker_dict = {ticker: price_currency + '_' + base_currency + '.FX' if base_currency != price_currency else None for ticker, base_currency in ticker_currency_dict.items()}
        unique_fx_ticker_list = list(set(ticker_fx_ticker_dict.values()))
        try:
            # if None is not in the list python raises a ValueError -> list.remove(x): x not in list
            unique_fx_ticker_list.remove(None)
        except ValueError:
            pass
        if len(unique_fx_ticker_list) == 0:
            return values_df
        fx_total_df = self.get_close_price_df(unique_fx_ticker_list)
        fx_quote_for_each_ticker_df = select_rows_from_dataframe_based_on_sub_calendar(fx_total_df, values_df.index)
        fx_quote_for_each_ticker_df = pd.DataFrame\
            (
                {
                    ticker: fx_quote_for_each_ticker_df.loc[:, ticker_fx_ticker_dict[ticker]]
                    if ticker_fx_ticker_dict[ticker] is not None
                    else 1  # in case the base currency is the same as the price currency
                    for ticker in list(values_df)
                }
            )
        return values_df.mul(fx_quote_for_each_ticker_df)

    def get_open_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(OpenPrice, tickers, start_date, end_date, currency)

    def get_high_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(HighPrice, tickers, start_date, end_date, currency)

    def get_low_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(LowPrice, tickers, start_date, end_date, currency)

    def get_close_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                           currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(ClosePrice, tickers, start_date, end_date, currency)

    def get_volume_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(Volume, tickers, start_date, end_date, currency)

    def get_liquidity_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                         currency: str = None):
        logger.debug('Get liquidity data' + logger_time_interval_message(start_date, end_date))
        close_price_df = self.get_close_price_df(tickers, start_date=start_date, end_date=end_date, currency=currency)
        volume_df = self.get_volume_df(tickers, start_date=start_date, end_date=end_date)
        volume_df = select_rows_from_dataframe_based_on_sub_calendar(volume_df, close_price_df.index)
        liquidity_df = close_price_df.multiply(volume_df)
        return liquidity_df

    def get_dividend_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                        currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(Dividend, tickers, start_date, end_date, currency)

    def get_total_return_df(self, tickers: {str, list}, start_date: datetime = None, end_date: datetime = None,
                            withholding_tax: {float} = 0.0, currency: str = None):
        logger.debug('Get total return data' + logger_time_interval_message(start_date, end_date))
        dividends = self.get_dividend_df(tickers, start_date=start_date, end_date=end_date)
        if dividends.empty:
            logger.info('No dividends paid' + logger_time_interval_message(start_date, end_date)
                        + 'Returning close price.')
            return self.get_close_price_df(tickers, start_date=start_date, end_date=end_date, currency=currency)
        else:
            close_price_local_ccy = self.get_close_price_df(tickers, start_date=start_date, end_date=end_date)
            close_price_roll_if_na = close_price_local_ccy.fillna(method='ffill')
            dividend_yield = dividends.divide(close_price_roll_if_na.shift()) * (1.0 - withholding_tax)
            dividend_yield = dividend_yield.loc[close_price_local_ccy.index]  # same index as the price DataFrame
            close_price_return = close_price_roll_if_na.pct_change()
            total_return = close_price_return + dividend_yield.fillna(value=0)
            cum_total_return = (1.0 + total_return.fillna(value=0)).cumprod()
            index_first_non_nan = close_price_roll_if_na.notna().idxmax()  # index of first non-NaN for each column
            first_value = np.diag(close_price_local_ccy.loc[index_first_non_nan])  # get the first non-NaN for each column
            cum_total_return *= first_value  # to have the same initial value as the original DataFrame
            cum_total_return += close_price_local_ccy * 0.0  # have NaN at same places as original DataFrame
            # convert the total return series into another currency if applicable.
            if isinstance(currency, str):
                return self._currency_convert_df(cum_total_return, currency)
            else:
                return cum_total_return

    def __repr__(self):
        return f"<FinancialDatabase(name = {self.database_name})>"


class _DynamicFinancialDatabase(FinancialDatabase):
    """Class definition for _DynamicFinancialDatabase. Adds functionality to create and update data."""

    def add_underlying(self, tickers: {str, list}) -> None:
        """Assumes that ticker is either a string or a list of strings. For each ticker the script downloads the
        information of the underlying and its data and inserts it to the database."""
        tickers = self.handle_ticker_input(tickers, convert_to_list=True)
        original_tickers = tickers.copy()
        logger.info("Attempts to populate the database with {} ticker(s).\nTicker: {}".format(len(original_tickers),
                                                                                              original_tickers))
        # Remove the tickers that already exists
        tickers = [ticker for ticker in tickers if not self.underlying_exist(ticker)]
        if len(tickers) == 0:
            logger.info('All {} ticker(s) already exist in the database.'.format(len(original_tickers)))
            return
        elif len(tickers) < len(original_tickers):  # if some tickers already existed in the database
            tickers_that_already_exist = list(set(original_tickers).difference(tickers))
            tickers_that_already_exist.sort()
            logger.info('{} ticker(s) already exists.\nTicker: {}'.format(len(tickers_that_already_exist),
                                                                          tickers_that_already_exist))
        logger.info("Populate the database with {} new ticker(s).\nTicker: {}".format(len(tickers), tickers))

        self._populate_underlying_table(tickers)
        self.refresh_data_for_tickers(tickers)
        return

    def refresh_data_for_tickers(self, tickers: {str, list}) -> None:
        """Assumes that tickers is either a string or list of strings. Refreshes the OHLC, Volume and dividend data up
        to today."""
        tickers = self.handle_ticker_input(tickers, convert_to_list=True)
        for ticker in tickers:  # check for existence. Only add or refresh data if ticker already exists
            if not self.underlying_exist(ticker):
                raise ValueError("{} does not exist in the database.\nUse 'add_underlying(<ticker>) to add it to the "
                                 "database'")
        tickers = self._control_tickers_before_refresh(tickers)  # some tickers might be de-listed...
        if len(tickers) == 0:  # no tickers to refresh or add data to
            return

        ticker_latest_obs_date_dict = self.get_ticker_underlying_attribute_dict(tickers,
                                                                                Underlying.latest_observation_date_with_values)
        if datetime.today().weekday() < 5:  # 0-6 represent the consecutive days of the week, starting from Monday.
            end_date = date.today()  # weekday
        else:
            end_date = date.today() - BDay(1)  # previous business day
        if None in ticker_latest_obs_date_dict.values():  # True if a ticker has no data: Load entire history!
            start_date = None
        else:
            start_date = min(ticker_latest_obs_date_dict.values())
            self._delete_open_high_low_close_volume_dividend_data(tickers, start_date, end_date)
        self._refresh_dividends(tickers, start_date, end_date)  # refresh dividends
        self._refresh_open_high_low_close_volume(tickers, start_date, end_date)  # refresh OHLC and Volume
        self._update_obs_date_and_dividend_history_status(tickers)  # update the dates
        return

    def _control_tickers_before_refresh(self, tickers: list, number_of_nan_days_threshold: int = 14) -> list:
        """Assumes tickers is a list of strings and number_of_nan_days_threshold is an int. Returns a new list of tickers
        (strings) where each ticker has no more than number_of_nan_days_threshold days between last observation date and
        last observation date with values. Also if the last observation date WITH VALUES equals today, remove the ticker
        since there is no need to refresh."""
        last_obs_date_dict = self.get_ticker_underlying_attribute_dict(tickers, Underlying.latest_observation_date)
        last_obs_date_with_values_dict = self.get_ticker_underlying_attribute_dict(tickers,
                                                                                   Underlying.latest_observation_date_with_values)
        if datetime.today().weekday() < 5:  # 0-6 represent the consecutive days of the week, starting from Monday.
            end_date = date.today()  # weekday
        else:
            end_date = date.today() - BDay(1)  # previous business day
        for ticker in tickers.copy():
            if not self.underlying_exist(ticker):
                num_nan_days = (last_obs_date_with_values_dict[ticker] - last_obs_date_dict[ticker]).days
                if last_obs_date_with_values_dict[ticker] == end_date:
                    tickers.remove(ticker)
                    logger.info("{} is up-to-date.".format(ticker))
                elif num_nan_days > number_of_nan_days_threshold:
                    tickers.remove(ticker)
                    logger.warning("{} has not published a value for the past {} days.".format(ticker, num_nan_days))
        if len(tickers) == 0:
            logger.info("All tickers are either up-to-date or not active.")
        return tickers

    def _populate_underlying_table(self, ticker_list: list):
        """Populates the Underlying table with rows based on the provided list of tickers. This method will be
        overridden in sub classes depending on the API used (e.g. Yahoo finance or Bloomberg)."""
        raise TypeError('_populate_underlying_table should not be called by an instance of a '
                        '_DynamicFinancialDatabase object')

    def _refresh_dividends(self, ticker_list: list, start_date: {date, datetime}=None,
                           end_date: {date, datetime}=None) -> None:
        """Populate the dividend table with ex-dividend dates and dividend amounts."""
        logger.debug('Refresh dividends for {} ticker(s)'.format(len(ticker_list))
                     + logger_time_interval_message(start_date, end_date))
        dividend_df = self._retrieve_dividend_df(ticker_list, start_date, end_date)
        logger.debug('Append rows to the Dividend table in the database.')
        dividend_df.to_sql(Dividend.__tablename__, self._session.get_bind(), if_exists='append', index=False)
        logger.debug('Commit the new Dividend rows.')
        self.session.commit()
        return

    def _retrieve_dividend_df(self, ticker_list: list, start_date: {date, datetime}, end_date: {date, datetime}):
        raise TypeError('_get_dividend_df should not be called by an instance of a _DynamicFinancialDatabase '
                        'object')

    def _refresh_open_high_low_close_volume(self, ticker_list: list, start_date: {date, datetime}=None,
                                            end_date: {date, datetime}=None) -> None:
        """Populate the OpenPrice, HighPrice, LowPrice, ClosePrice and Volume tables with new rows."""
        logger.debug('Refresh OHLC and volume for {} ticker(s)'.format(len(ticker_list))
                     + logger_time_interval_message(start_date, end_date))
        open_high_low_close_volume_df = self._retrieve_open_high_low_close_volume_df(ticker_list, start_date, end_date)

        data_type_list = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        value_name_list = [name.lower() + '_quote' for name in data_type_list]
        table_name_list = [OpenPrice.__tablename__, HighPrice.__tablename__, LowPrice.__tablename__,
                           ClosePrice.__tablename__, Volume.__tablename__]
        for i in range(len(data_type_list)):
            logger.debug("Append rows to the '{}' table in the database.".format(table_name_list[i]))
            value_df = open_high_low_close_volume_df[open_high_low_close_volume_df['data_type'] == data_type_list[i]].copy()
            value_df.drop('data_type', axis=1, inplace=True)
            value_df.rename(columns={'value': value_name_list[i]}, inplace=True)
            value_df.to_sql(table_name_list[i], self.session.get_bind(), if_exists='append', index=False)
        logger.debug('Commit the new OHLC and Volume rows.')
        self.session.commit()
        return

    def _retrieve_open_high_low_close_volume_df(self, ticker_list: list, start_date: {date, datetime},
                                                end_date: {date, datetime}) -> pd.DataFrame:
        """Should return a DataFrame with 'data_type', 'obs_date', 'value', 'comment', 'data_source', 'underlying_id' as
        column headers"""
        raise TypeError('_get_open_high_low_close_volume_df should not be called with an instance of a '
                        '_DynamicFinancialDatabase object')

    def __repr__(self):
        return f"<_DynamicFinancialDatabase(name = {self.database_name})>"


class YahooFinancialDatabase(_DynamicFinancialDatabase):
    """Class definition of YahooFinancialDatabase
    Using the Yahoo Finance API, this class an add and create data to the database."""

    def _populate_underlying_table(self, ticker_list: list) -> None:
        yf_ticker_list = self.yahoo_finance_ticker(ticker_list)
        underlying_list = []
        counter = 0
        for yf_ticker in yf_ticker_list:
            try:
                progression_bar(counter + 1, len(yf_ticker_list))
                logger.debug('Fetching data dictionary from Yahoo Finance for {}...'.format(yf_ticker.ticker))
                ticker_info = yf_ticker.info  # retrieves a dictionary with data e.g. name and sector (takes a while)
            except KeyError:
                raise ValueError("'{}' does not exist as a ticker on Yahoo Finance.".format(yf_ticker.ticker))
            default_str = 'NA'  # in case the attribute does not exist, use a default string
            # e.g. it is normal for an INDEX or FX rate to not have a website
            underlying = Underlying(ticker=ticker_list[counter],
                                    underlying_type=ticker_info.get('quoteType', default_str),
                                    long_name=ticker_info.get('longName'),
                                    short_name=ticker_info.get('shortName'),
                                    sector=capital_letter_no_blanks(ticker_info.get('sector', default_str)),
                                    industry=capital_letter_no_blanks(ticker_info.get('industry', default_str)),
                                    currency=capital_letter_no_blanks(ticker_info.get('currency', default_str)),
                                    country=capital_letter_no_blanks(ticker_info.get('country', default_str)),
                                    city=ticker_info.get('city', default_str),
                                    address=ticker_info.get('address1', default_str),
                                    description=ticker_info.get('longBusinessSummary', default_str),
                                    web_site=ticker_info.get('website', default_str))
            underlying_list.append(underlying)
            counter += 1
        logger.debug('Append {} row(s) to the Underlying table in the database.'.format(len(underlying_list)))
        self.session.add_all(underlying_list)
        logger.debug('Commit the new Underlying rows.')
        self.session.commit()
        return

    def _retrieve_dividend_df(self, ticker_list: list, start_date: {date, datetime}, end_date: {date, datetime}) \
            -> pd.DataFrame:
        logger.debug("Downloading dividend data from Yahoo Finance and reformat the DataFrame.")
        yf_ticker_list = self.yahoo_finance_ticker(ticker_list)  # need to download the dividends per YF ticker
        dividend_amount_total_df = None  # initialize the resulting DataFrame.
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        counter = 0
        for yf_ticker in yf_ticker_list:  # loop through each Yahoo Finance Ticker object
            if start_date is None:
                yf_historical_data_df = yf_ticker.history(period='max')  # the maximum available data
            else:
                yf_historical_data_df = yf_ticker.history(start=start_date, end=end_date + timedelta(days=1))
            # yf_historical_data_df contains Open, High, Low, Close, Volume, Dividends and Stock Splits
            dividend_df = yf_historical_data_df.reset_index()[['Date', 'Dividends']]  # extract dates and dividend
            dividend_amount_df = dividend_df[dividend_df['Dividends'] != 0]  # remove the rows with zero. Now the dates
            # are the ex-dividend dates
            dividend_amount_df.insert(0, 'underlying_id', ticker_underlying_id_dict[ticker_list[counter]])
            if dividend_amount_total_df is None:
                dividend_amount_total_df = dividend_amount_df.copy()
            else:
                # for each ticker, combine the DataFrames
                dividend_amount_total_df = pd.concat([dividend_amount_total_df, dividend_amount_df], ignore_index=True)
            counter += 1
        # add comment, name of data source and rename and reshuffle the columns
        dividend_amount_total_df['comment'] = 'Loaded at {}'.format(str(date.today()))
        dividend_amount_total_df['data_source'] = 'YAHOO_FINANCE'
        dividend_amount_total_df.rename(columns={'Date': 'ex_div_date', 'Dividends': 'dividend_amount'}, inplace=True)
        return dividend_amount_total_df[['ex_div_date', 'dividend_amount', 'comment', 'data_source', 'underlying_id']]

    def _retrieve_open_high_low_close_volume_df(self, ticker_list: list, start_date: {date, datetime},
                                                end_date: {date, datetime}) -> pd.DataFrame:
        logger.debug("Downloading OHLC and volume data from Yahoo Finance and reformat the DataFrame.")
        multiple_ticker_str = self.multiple_ticker_string(ticker_list)  # ['ABC', 'DEF'] -> 'ABC DEF'
        yf_historical_data_df = yfinance.download(tickers=multiple_ticker_str, start=start_date,
                                                  end=end_date + timedelta(days=1))  # need to add an extra date
        yf_historical_data_df = yf_historical_data_df.loc[start_date:]
        # E.g. using the two tickers 'ABB.ST' and 'MCD' the DataFrame yf_historical_data_df has the below shape:
        #              Adj Close              ...     Volume
        #                 ABB.ST         MCD  ...     ABB.ST        MCD
        # Date                                ...
        # 1966-07-05         NaN    0.002988  ...        NaN   388800.0
        # 1966-07-06         NaN    0.003148  ...        NaN   687200.0
        # 1966-07-07         NaN    0.003034  ...        NaN  1853600.0
        yf_historical_data_unstacked_df = yf_historical_data_df.unstack().reset_index()
        if len(ticker_list) == 1:  # add the ticker if there is only one ticker (gets removed by default)
            yf_historical_data_unstacked_df.insert(loc=1, column='ticker', value=ticker_list[0])
        # convert back the tickers from Yahoo Finance format
        # first create a dictionary with tickers with and without the Yahoo Finance format
        ticker_yahoo_finance_format_dict = dict(zip(self.convert_ticker_to_yahoo_finance_format(ticker_list), ticker_list))
        # remove the tickers that does not need replacement
        ticker_yahoo_finance_format_dict = {key: value for key, value in ticker_yahoo_finance_format_dict.items()
                                            if key != value}
        yf_historical_data_unstacked_df.replace({'ticker': ticker_yahoo_finance_format_dict})
        yf_historical_data_unstacked_df.columns = ['data_type', 'ticker', 'obs_date', 'value']

        # add comment and name of data source
        yf_historical_data_unstacked_df['comment'] = 'Loaded at {}'.format(str(date.today()))
        yf_historical_data_unstacked_df['data_source'] = 'YAHOO_FINANCE'

        # replace the tickers with the corresponding underlying id
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        yf_historical_data_unstacked_df['underlying_id'] = \
            yf_historical_data_unstacked_df['ticker'].apply(lambda ticker: ticker_underlying_id_dict[ticker])
        yf_historical_data_unstacked_df.drop(['ticker'], axis=1, inplace=True)

        # clean up by removing NaN and zeros
        yf_historical_data_unstacked_clean_df = yf_historical_data_unstacked_df[
            pd.notnull(yf_historical_data_unstacked_df['value'])].copy()  # remove rows with NaN in data column
        yf_historical_data_unstacked_clean_df = yf_historical_data_unstacked_clean_df[
            yf_historical_data_unstacked_clean_df['value'] != 0].copy()  # remove rows with 0 in data column
        yf_historical_data_unstacked_clean_df['data_type'] = yf_historical_data_unstacked_clean_df['data_type'].str.upper()
        return yf_historical_data_unstacked_clean_df[['data_type', 'obs_date', 'value', 'comment', 'data_source',
                                                      'underlying_id']]

    @staticmethod
    def convert_ticker_to_yahoo_finance_format(ticker_list: list):
        """Assumes that ticker_list is a list of strings. Returns a new list where each ticker in list has been adjusted
        to take into account the FX ticker format (base currency_price currency.FX) and index ticker format (name.INDEX)
        """
        if not isinstance(ticker_list, list):
            raise TypeError('ticker_list needs to be a list.')
        adjusted_ticker_list = []
        for ticker in ticker_list:
            if ticker.endswith('.FX'):  # check if ticker has the FX suffix
                ticker = ticker.replace('.FX', '')  # remove the suffix
                if ticker.endswith('_USD'):  # in Yahoo Finance, if the base currency is USD, the 'USD' is omitted
                    ticker = ticker.replace('_USD', '')
                else:
                    ticker = ticker.split('_')[1] + ticker.split('_')[0]
                    # ticker = ticker.replace('_', '')
                ticker += '=X'  # add the Yahoo Finance FX suffix
            elif ticker.endswith('.INDEX'):
                ticker = '^' + ticker.replace('.INDEX', '')
            else:
                pass  # do nothing to the ticker
            adjusted_ticker_list.append(ticker)
        return adjusted_ticker_list

    def yahoo_finance_ticker(self, tickers: list):
        adjusted_ticker_list = self.convert_ticker_to_yahoo_finance_format(tickers)
        # create a list of Yahoo Finance ticker objects
        yf_tickers = [yfinance.Ticker(ticker) for ticker in adjusted_ticker_list]
        return yf_tickers

    def multiple_ticker_string(self, ticker_list: list) -> str:
        """Assumes that ticker_list is a list of strings. Method returns a string containing each ticker as a
        substring. E.g. the list ['TICKER_A', 'TICKER_B'] yields 'TICKER_A TICKER_B'"""
        adjusted_ticker_list = self.convert_ticker_to_yahoo_finance_format(ticker_list)
        result_string = ''
        for ticker in adjusted_ticker_list:
            result_string += str(ticker)
            if ticker != adjusted_ticker_list[-1]:
                result_string += ' '
        return result_string

    def __repr__(self):
        return f"<YahooFinancialDatabase(name = {self.database_name})>"


def logger_time_interval_message(start_date: {date, datetime}, end_date: {date, datetime}) -> str:
    logger_message = ''
    if start_date is not None:
        logger_message += ' from {}'.format(str(start_date)[:10])
    if end_date is not None:
        logger_message += ' up to {}'.format(str(end_date)[:10])
    logger_message += '.'
    return logger_message

