"""
automatic_database_refresh.py
"""
from datetime import datetime, time

from financial_database import YahooFinanceFeeder
from config_database import my_database_name, base_folder
from models_db import Underlying

from sms import send_sms

import logging

__TOKYO_CLOSE__ = time(hour=15, minute=0, second=0)
__STOCKHOLM_CLOSE__ = time(hour=1, minute=30, second=0)  # T + 1 TK time
__NEW_YORK_CLOSE__ = time(hour=5, minute=0, second=0)  # T + 1 TK time
# time zones: https://www.timeanddate.com/worldclock/converter.html?iso=20200328T020000&p1=248&p2=179&p3=239
# exchange opening hours: https://en.wikipedia.org/wiki/List_of_stock_exchange_trading_hours

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
log_file_name = base_folder + '\\automatic_refresh'
file_handler = logging.FileHandler(log_file_name + '.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


def refresh_tickers():
    """ Based on the time of day, refresh certain selection of tickers. All times are with respect to Tokyo time
    time zones: https://www.timeanddate.com/worldclock/converter.html?iso=20200328T020000&p1=248&p2=179&p3=239
    exchange opening hours: https://en.wikipedia.org/wiki/List_of_stock_exchange_trading_hours
    """
    now = datetime.now()
    ticker_filter = {Underlying.underlying_type: ['EQUITY', 'INDEX'], Underlying.currency: ''}
    exchange_closings_ccy_dict = {__TOKYO_CLOSE__: 'JPY',
                                  __STOCKHOLM_CLOSE__: 'SEK',
                                  __NEW_YORK_CLOSE__: 'USD'}

    if now.time() <= min(exchange_closings_ccy_dict.keys()):
        currency = exchange_closings_ccy_dict[max(exchange_closings_ccy_dict.keys())]
    else:
        minutes_after_closing = [number_of_minutes(now.time()) - number_of_minutes(exchange_close)
                                 for exchange_close in exchange_closings_ccy_dict.keys()]
        # pick the minimum non negative element from 'minutes_after_closing'
        min_non_negative_minutes_after_closing = min(x for x in minutes_after_closing if x >= 0)
        exchange_closings = list(exchange_closings_ccy_dict.keys())
        exchange_closing = exchange_closings[minutes_after_closing.index(min_non_negative_minutes_after_closing)]
        currency = exchange_closings_ccy_dict[exchange_closing]

    # add the correct currency to the filter
    ticker_filter[Underlying.currency] = currency

    # create a database handler and refresh the eligible tickers
    fin_db_handler = YahooFinanceFeeder(my_database_name)
    tickers = fin_db_handler.get_ticker(ticker_filter)
    try:
        fin_db_handler.refresh_data_for_tickers(tickers)
        logger.debug('Successfully refreshed data for {} ticker(s) denominated in {}.'.format(len(tickers), currency))
    except Exception:
        logger.error("Error during refresh denominated in {}.".format(currency), exc_info=True)
        send_sms('Error during refresh denominated in {}.'.format(currency))


def number_of_minutes(time_: time) -> int:
    """
    Returns the number of total minutes
    :param time_: time
    :return: int
    """
    return time_.hour * 60 + time_.minute


if __name__ == '__main__':
    refresh_tickers()

