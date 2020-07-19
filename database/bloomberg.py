"""
bloomberg.py
"""
try:
    import pdblp
except ModuleNotFoundError:
    print("No Bloomberg acces found (when importing 'pdblp')")

try:
    from xbbg import blp
except ModuleNotFoundError:
    print("No Bloomberg acces found (when importing 'xbbg')")

import pandas as pd
from datetime import datetime
import logging

# my modules
from general_tools import list_grouper, progression_bar

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class BloombergConnection:

    batch_size = 100

    def __init__(self, use_debug: bool=True):
        self.con = pdblp.BCon(debug=use_debug, port=8194, timeout=10000)
        self.con.start()
        self.default_start_date = '19500101'

    def reconnect(self):
        logger.warning('Reconnecting to Bloomberg due to run time error.')
        self.con.start()

    def get_last_price(self, tickers: {str, list}, start_date: datetime = None, end_date: datetime = None):
        """
        Get a DataFrame with close prices (PX_LAST) for the given ticker(s) between the given dates.
        :param tickers: can be a string or list of strings
        :param start_date: datetime.datetime. If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: datetime.datetime. If None, the end date will be set to today
        :return: DataFrame
        """
        last_price_bbg_df = self._get_daily_data(tickers, 'PX_LAST', start_date, end_date)
        return last_price_bbg_df

    def get_volume(self, tickers: {str, list}, start_date: datetime = None, end_date: datetime = None):
        """
        Get a DataFrame with volumes (PX_VOLUME) for the given ticker(s) between the given dates.
        :param tickers: can be a string or list of strings
        :param start_date: datetime.datetime. If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: datetime.datetime. If None, the end date will be set to today
        :return: DataFrame
        """
        last_price_bbg_df = self._get_daily_data(tickers, 'PX_VOLUME', start_date, end_date)
        return last_price_bbg_df

    def get_daily_data(self, tickers: {str, list}, field: {str, list}, start_date: datetime = None, end_date: datetime = None):
        """
        Get a containing data for the given field(s) and ticker(s) between the given dates.
        :param tickers: can be a string or list of strings
        :param start_date: datetime.datetime. If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: datetime.datetime. If None, the end date will be set to today
        :return: DataFrame
        """
        last_price_bbg_df = self._get_daily_data(tickers, field, start_date, end_date)
        return last_price_bbg_df

    def _get_daily_data(self, tickers: {str, list}, field: {str, list}, start_date: datetime = None, end_date: datetime = None):
        """
        Get a containing data for the given field(s) and ticker(s) between the given dates.
        :param tickers: can be a string or list of strings
        :param start_date: datetime.datetime. If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: datetime.datetime. If None, the end date will be set to today
        :return: DataFrame
        """
        # adjust inputs
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = self.add_bbg_ticker_suffix(tickers)
        if end_date is None:
            end_date = datetime.today()

        # logger
        if type(field) == str:
            logger.debug('Downloading {} data from Bloomberg.'.format(field.upper().replace(' ', '_')))
        else:
            field = [fld.upper().replace(' ', '_') for fld in field]
            logger.debug('Downloading %s data from Bloomberg.' % ', '.join(field))

        # loop through the batches of tickers
        daily_data_bbg_df = None
        ticker_batches = list_grouper(tickers, self.batch_size)
        counter = 1
        for ticker_sub_list in ticker_batches:
            progression_bar(counter, len(ticker_batches))

            # get the data from Bloomberg
            while True:
                try:
                    sub_daily_data_bbg_df = self.con.bdh(ticker_sub_list, field, self.bbg_date(start_date), self.bbg_date(end_date))
                    break
                except RuntimeError:
                    self.reconnect()
                    sub_daily_data_bbg_df = self.con.bdh(ticker_sub_list, field, self.bbg_date(start_date), self.bbg_date(end_date))

            if daily_data_bbg_df is None:
                daily_data_bbg_df = sub_daily_data_bbg_df
            else:
                daily_data_bbg_df = pd.concat([daily_data_bbg_df, sub_daily_data_bbg_df], join='outer', axis=1)
            counter += 1
        if isinstance(field, str):
            daily_data_bbg_df.columns = daily_data_bbg_df.columns.droplevel(1)  # remove the field name
        return daily_data_bbg_df

    def get_dividend_data(self, tickers: {str, list}, start_date: datetime = None, end_date: datetime = None, do_pivot: bool = True):
        """
        Get a DataFrame containing dividend amount for the given tickers between the given dates (ex-dividend dates)
        :param tickers: can be a string or list of strings
        :param start_date: datetime.datetime. If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: datetime.datetime. If None, the end date will be set to today
        :param do_pivot: if true, DataFRame is pivoted: index = ex-dividend date, values = dividend amount, columns = tickers
        :return: DataFrame
        """
        # adjust inputs
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = self.add_bbg_ticker_suffix(tickers)
        if end_date is None:
            end_date = datetime.today()

        logger.debug('Downloading dividend data from Bloomberg.')

        # loop through the batches of tickers
        dividend_bbg_df = None
        ticker_batches = list_grouper(tickers, self.batch_size)
        counter = 1
        for ticker_sub_list in ticker_batches:
            progression_bar(counter, len(ticker_batches))
            # get the data from Bloomberg
            while True:
                try:
                    sub_dividend_bbg_df = blp.bds(ticker_sub_list, 'DVD_HIST', 'Header=Y')
                    break
                except RuntimeError:
                    self.reconnect()
                    sub_dividend_bbg_df = blp.bds(ticker_sub_list, 'DVD_HIST', 'Header=Y')
            try:
                sub_dividend_bbg_df = sub_dividend_bbg_df[['ex_date', 'dividend_amount']].copy()
            except KeyError:  # in case there are no dividends
                pass
            else:
                sub_dividend_bbg_df.reset_index(inplace=True)
                if dividend_bbg_df is None:
                    dividend_bbg_df = sub_dividend_bbg_df
                else:
                    dividend_bbg_df = pd.concat([dividend_bbg_df, sub_dividend_bbg_df], ignore_index=True, sort=False)
            counter += 1
        if dividend_bbg_df is None:
            return

        # only include eligible ex dividend dates
        dividend_bbg_df = self._adjust_dividend_df(tickers, dividend_bbg_df, start_date, end_date, do_pivot)
        return dividend_bbg_df

    @staticmethod
    def _adjust_dividend_df(tickers: list, dividend_df: pd.DataFrame, start_date: {datetime, None}, end_date: {datetime, None}, do_pivot: bool):
        """
        Adjusts and re-formats when applicable a DataFrame with dividend data
        :param tickers: can be a string or list of strings
        :param dividend_df: DataFrame with dividend data
        :param start_date: If None, the start date will be set to a default date (e.g. 1 jan 1950)
        :param end_date: If None, the end date will be set to today
        :param do_pivot: If true, DataFRame is pivoted: index = ex-dividend date, values = dividend amount, columns = tickers
        :return: DataFrame
        """
        # only include eligible ex dividend dates
        dividend_df.sort_values(by=['ex_date'], inplace=True)  # sort the dates
        dividend_df['ex_date'] = pd.to_datetime(dividend_df['ex_date'])  # convert the dates to datetime
        if start_date is not None:
            dividend_df = dividend_df[dividend_df['ex_date'] >= start_date]
        if end_date is not None:
            dividend_df = dividend_df[dividend_df['ex_date'] <= end_date]

        # for each ticker sum dividend payments on same ex-dates
        dividend_df = dividend_df.groupby(by=['ticker', 'ex_date'])['dividend_amount'].sum()
        dividend_df.reset_index(inplace=True)

        if do_pivot:
            dividend_df = pd.pivot_table(dividend_df, values='dividend_amount', index='ex_date', columns='ticker')
            # add back ticker(s) that does not pay any dividends
            missing_tickers = list(set(tickers).difference(list(dividend_df)))
            dividend_df = dividend_df.reindex(columns=list(dividend_df) + missing_tickers)
            dividend_df = dividend_df[tickers]
        return dividend_df

    def get_underlying_information(self, tickers: {str, list}, field: {str, list}):
        """
        Returns a DataFrame with underlying information e.g. sector names for the given ticker(s)
        :param tickers: van be a string or list of strings
        :param field: non-case sensitive string (e.g. GICS_SECTOR_NAME)
        :return: DataFrame
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = self.add_bbg_ticker_suffix(tickers)
        if isinstance(field, str):
            field = [field]
        field = [fld.upper().replace(' ', '_') for fld in field]  # capital cases and replace blanks with underscore

        while True:
            try:
                underlying_data_bbg_df = self.con.ref(tickers, field)
                break
            except RuntimeError:
                self.reconnect()
                underlying_data_bbg_df = self.con.ref(tickers, field)

        underlying_data_bbg_pivot_df = pd.pivot_table(underlying_data_bbg_df, values='value', index='ticker',
                                                      columns='field', aggfunc=lambda x: ' '.join(str(v) for v in x))
        return underlying_data_bbg_pivot_df

    def get_index_members(self, index_ticker: str, observation_date: {datetime, list}=None,
                          add_bbg_suffix: bool = True) -> {list, dict}:
        """
        Returns a list or a dictionary (key = datetime, values = list of tickers) of index members for the given ticker
        and the observation date(s). If Observation dates are not given, script returns the index members as of today.
        :param index_ticker: string
        :param observation_date: datetime or list of datetime
        :param add_bbg_suffix: bool
        :return: list of strings or dictionary
        """
        # handle various observation date inputs
        if observation_date is None:
            observation_date = datetime.today()
        elif isinstance(observation_date, pd.DatetimeIndex):
            observation_date = [obs_date for obs_date in observation_date]
        if type(observation_date) != list:
            return_dict = False
            observation_date = [observation_date]
        else:
            return_dict = True
        result_dict = {}
        for obs_date in observation_date:
            logger.debug('Donwloading historical index members of {} observed at {} from Bloomberg.'.format(index_ticker.upper(), obs_date))
            bbg_obs_date = self.bbg_date(obs_date)
            bulk_data_bbg = self.con.bulkref(index_ticker, 'INDX_MWEIGHT_HIST', [('END_DATE_OVERRIDE', bbg_obs_date)])
            index_members = bulk_data_bbg[bulk_data_bbg['name'] == 'Index Member']['value'].values
            if add_bbg_suffix:
                index_members = self.add_bbg_ticker_suffix(index_members)
            result_dict.update({obs_date: index_members})
        if return_dict:
            return result_dict
        else:
            return result_dict[observation_date[0]]

    def get_index_inclusion_df(self, index_ticker: str, observation_calendar: pd.DatetimeIndex,
                               add_bbg_suffix: bool = True) -> pd.DataFrame:
        """
        Returns a DataFrame with tickers in alphabetic order as column headers and observation dates as index. Vaue is 1
        if the ticker is included in the index for that particular observation date, else 0.
        :param index_ticker: str
        :param observation_calendar: pd.DatetimeIndex
        :param add_bbg_suffix: bool
        :return: pd.DataFrame
        """
        obs_date_ticker_list_dict = self.get_index_members(index_ticker, observation_calendar, add_bbg_suffix)
        
        # get all the unique tickers and sort in alphabetic order
        tickers = []
        for ticker_list in obs_date_ticker_list_dict.values():
            tickers.extend(ticker_list)
        tickers = list(set(tickers))
        tickers.sort()
        
        # loop through all observation dates and set value to 1 if ticker (column) exists in the index, else 0.
        result_df = pd.DataFrame(columns=tickers)
        for obs_date in observation_calendar:
            tickers_in_index = obs_date_ticker_list_dict[obs_date]
            ticker_inclusivity = [1 if ticker in tickers_in_index else 0 for ticker in tickers]
            result_df.loc[obs_date] = ticker_inclusivity

        # remove the first rows where all columns are zero
        cum_num_tickers = result_df.sum(axis=1).cumsum()
        eligible_index = cum_num_tickers[cum_num_tickers > 0].index
        return result_df.loc[eligible_index, :]

    def get_futures_chain(self, generic_futures_index_ticker: str):
        """
        Returns a list of tickers of underlying futures contracts that are part of a generic futures index
        :param generic_futures_index_ticker: string
        :return: list of strings
        """
        bulk_data_bbg = self.con.bulkref(generic_futures_index_ticker, 'FUT_CHAIN_LAST_TRADE_DATES',
                                         [('INCLUDE_EXPIRED_CONTRACTS', 'Y')])
        bulk_data_bbg = bulk_data_bbg[bulk_data_bbg['name'] == "Future's Ticker"]
        futures_tickers = list(bulk_data_bbg['value'].values)
        return futures_tickers

    # TODO method to load fundamental data from companies
    def get_fundamental_data(self):
        pass

    def bbg_date(self, input_date) -> str:
        """
        Converts the date to Bloomberg format: 'YYYYMMDD'
        """
        if input_date is None:
            return self.default_start_date
        input_date_str = str(input_date)
        day = input_date_str[8:10]
        month = input_date_str[5:7]
        year = input_date_str[:4]
        return year + month + day

    def bbg_today(self):
        return self.bbg_date(datetime.today())

    @staticmethod
    def convert_bbg_date_to_date(bbg_date: str)->datetime:
        """
        Converts the Bloomberg date (string) to datetime format
        :param bbg_date: str
        :return: datetime
        """
        day = bbg_date[6:8]
        month = bbg_date[4:6]
        year = bbg_date[:4]
        return datetime(int(year), int(month), int(day))

    @staticmethod
    def add_bbg_ticker_suffix(tickers: {str, list}, suffix: str = 'EQUITY') -> list:
        """
        Adds a suffix to all the tickers in the ticker list if applicable
        :param tickers: list of strings
        :param suffix: sub-string to be added to the original ticker
        :return: list of strings
        """
        return BloombergConnection._adjust_bbg_tickers(tickers, suffix, True)

    @staticmethod
    def remove_bbg_ticker_suffix(tickers: {str, list}, suffix: str = 'EQUITY')->list:
        """
        Removes a suffix from all the tickers in the ticker list if applicable
        :param tickers: list of strings
        :param suffix: sub-string to be removed from the original ticker
        :return: list of strings
        """
        return BloombergConnection._adjust_bbg_tickers(tickers, suffix, False)

    @staticmethod
    def _adjust_bbg_tickers(tickers: {str, list}, suffix: str, add_suffix: bool) -> list:
        """
        Removes (adds) a given suffix from (to) all the tickers in the ticker list if applicable
        :param tickers: list of strings
        :param suffix: sub-string to be removed from the original ticker
        :param add_suffix: if true, add suffix else remove
        :return: list of strings
        """
        is_string = type(tickers) == str
        if is_string:
            tickers = [tickers]
        adj_ticker_list = []
        suffix = suffix.upper()
        for ticker in tickers:
            ticker = ticker.upper()
            if add_suffix:
                has_suffix = ticker.endswith('EQUITY') or ticker.endswith('INDEX') or ticker.endswith(
                    'COMDTY') or ticker.endswith(suffix)
                if not has_suffix:
                    ticker += ' ' + suffix
            else:
                if ticker.endswith(suffix):
                    ticker = ticker.replace(' ' + suffix, '')
            adj_ticker_list.append(ticker)
        if is_string:
            return adj_ticker_list[0]
        else:
            return adj_ticker_list
