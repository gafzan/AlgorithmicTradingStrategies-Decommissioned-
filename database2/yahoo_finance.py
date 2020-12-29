"""
yahoo_finance.py
"""
import yfinance as yf
from datetime import date, timedelta
import pandas as pd
import numpy as np

import logging

from tools.general_tools import time_period_logger_msg, list_grouper

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# TODO handle ConnectionError?


class YahooFinanceConnection:

    def get_ohlc_volume(self, tickers: {str, list}, start_date: date = None, end_date: date = None) -> dict:
        """
        Returns a dictionary with DataFrames containing Adj Close, Open, High, Low, Close and Volume
        keys = 'Adj Close', 'Open', 'High', 'Low', 'Close' and 'Volume'
        values = DataFrames
        :param tickers: str, list
        :param start_date: date, None
        :param end_date: date, None
        :return: dict
        """
        # load the data from Yahoo Finance
        logger.debug('Loading OHLC & Volume data from Yahoo Finance' + time_period_logger_msg(start_date=start_date, end_date=end_date))
        end_date = end_date if end_date is None else end_date + timedelta(days=1)  # adjust since 'end' is not inclusive

        # download data in batches of 100 tickers each
        tickers = tickers if isinstance(tickers, list) else [tickers]
        yf_historical_data_df = None
        for ticker_sub_list in list_grouper(tickers, 100):
            yf_historical_data_sub_df = yf.download(tickers=self.get_multiple_ticker_str(tickers=ticker_sub_list),
                                                    start=start_date, end=end_date, auto_adjust=False)
            # E.g. using the two tickers 'ABB.ST' and 'MCD' the DataFrame yf_historical_data_df has the below shape:
            #              Adj Close              ...     Volume
            #                 ABB.ST         MCD  ...     ABB.ST        MCD
            # Date                                ...
            # 1966-07-05         NaN    0.002988  ...        NaN   388800.0
            if yf_historical_data_sub_df.columns.inferred_type != 'mixed':
                yf_historical_data_df.columns = pd.MultiIndex.from_product([yf_historical_data_df.columns, ticker_sub_list])

            if yf_historical_data_df is None:
                yf_historical_data_df = yf_historical_data_sub_df
            else:
                yf_historical_data_df = yf_historical_data_df.join(yf_historical_data_sub_df, how='outer')
        yf_historical_data_df = yf_historical_data_df[start_date:]  # handle case when start_date is a holiday
        result_dict = {}
        if yf_historical_data_df.columns.inferred_type != 'mixed':
            # add a second column layer in case only one ticker was provided
            ticker = tickers.upper() if isinstance(tickers, str) else tickers[0].upper()
            yf_historical_data_df.columns = pd.MultiIndex.from_product([yf_historical_data_df.columns, [ticker]])
        data_type_names = yf_historical_data_df.columns.levels[0].values
        for data_type_name in data_type_names:
            data = yf_historical_data_df.xs(data_type_name, axis=1, level=0, drop_level=True)
            data = data.dropna(how='all')
            data = self.reorder_tickers(df=data, tickers=tickers)
            result_dict.update({data_type_name: data})
        logger.debug('Done with loading OHLC & Volume data from Yahoo Finance')
        return result_dict

    def get_info_dict(self, tickers: {str, list}):
        """
        Returns a dictionary with tickers as keys and a sub-dictionary as values that are filled with information about
        the instrument
        :param tickers: str, list
        :return: dict {ticker: {info key: info value}}
        """
        yf_tickers = self.get_yf_tickers(tickers=tickers)
        info_dict = {}  # initialize the resulting dict
        num_tickers = len(yf_tickers)  # used in logger
        counter = 1  # used in logger
        for yf_ticker in yf_tickers:
            logger.info(f"Loading information dictionary ({counter}/{num_tickers})")
            info_dict = self._update_info_dict(yf_ticker=yf_ticker, info_dict=info_dict)
            counter += 1
        logger.debug('Done with loading instrument information from Yahoo Finance')
        return info_dict

    @staticmethod
    def _update_info_dict(yf_ticker, info_dict: dict)->dict:
        """
        Make some manual adjustments and additions to the information map
        :param yf_ticker:
        :param info_dict: dict
        :return: dict
        """
        try:
            sub_info_dict = yf_ticker.info
        except KeyError:
            raise ValueError("Can't load info for '{}'".format(yf_ticker.ticker))
        if sub_info_dict['quoteType'] in ['EQUITY', 'CURRENCY']:
            sub_info_dict['asset_class'] = sub_info_dict['quoteType']
        else:
            sub_info_dict['asset_class'] = None
        info_dict.update({yf_ticker.ticker: yf_ticker.info})
        return info_dict

    def get_dividend_and_splits(self, tickers: {str, list}, start_date: date = None, end_date: date = None) -> (pd.DataFrame, pd.DataFrame):
        """
        Returns a tuple with two DataFrames containing dividends and stock splits (assuming tickers as column names and
        dates as index)
        :param tickers: str, list
        :param start_date: date, None
        :param end_date: date, None
        :return: tuple (dividend DataFrame, split DataFrame)
        """
        # downloads dividends and splits, adjust the start and end dates and store the DataFrame in a list
        logger.debug('Loading Dividend & Stock Splits from Yahoo Finance' + time_period_logger_msg(start_date=start_date, end_date=end_date))
        yf_tickers = self.get_yf_tickers(tickers=tickers)
        corp_actions_df_list = []
        for yf_ticker in yf_tickers:
            corp_actions_df_list.append(yf_ticker.actions[start_date:end_date])

        # concatenate such that the final DataFrame has two layers of column names:
        # Level #1: ticker 1                  ticker 2                    ...
        # Level #2: Dividend  Stock Splits    Dividend    Stock Splits    ...
        total_corp_actions_df = pd.concat(dict(zip([yf_ticker.ticker for yf_ticker in yf_tickers], corp_actions_df_list)), axis=1)
        # select the dividend and stock splits separately
        div_df = total_corp_actions_df.xs('Dividends', axis=1, level=1, drop_level=True)
        split_df = total_corp_actions_df.xs('Stock Splits', axis=1, level=1, drop_level=True)

        # clean result by setting 0 to NaN and removing rows where all columns are NaN
        div_df.replace(0, np.nan, inplace=True)
        div_df = div_df.dropna(how='all')
        split_df.replace(0, np.nan, inplace=True)
        split_df = split_df.dropna(how='all')
        logger.debug('Done with loading Dividends & Stock Splits from Yahoo Finance')
        return {'Dividends': self.reorder_tickers(df=div_df, tickers=tickers),
                'Stock Splits': self.reorder_tickers(df=split_df, tickers=tickers)}

    def get_yf_tickers(self, tickers: {str, list}):
        """
        Returns an iterable of yfinance.Ticker objects
        :param tickers: str, list
        :return: yfinance.Tickers.tickers
        """
        return yf.Tickers(tickers=self.get_multiple_ticker_str(tickers)).tickers

    @staticmethod
    def get_multiple_ticker_str(tickers: {str, list}):
        return '%s' % " ".join(tickers) if isinstance(tickers, list) else tickers

    @staticmethod
    def reorder_tickers(df: pd.DataFrame, tickers: {str, list}):
        """
        Since the DataFrames from Yahoo Finance returns the result assuming alphabetic ordering of the tickers, this
        method is used to reverse that
        :param df: DataFrame
        :param tickers: str, list
        :return: DataFrame
        """
        # re-order to the original order of the tickers
        if isinstance(tickers, list):
            column_names = [ticker.upper() for ticker in tickers]
            return df[column_names]
        else:
            return df

