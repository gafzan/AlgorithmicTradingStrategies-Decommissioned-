from bs4 import BeautifulSoup
import requests
from time import sleep
from datetime import date
import pandas as pd
import logging

# my modules
from database.config_database import __EXCEL_TICKER_FOLDER__
from excel_tools import save_df

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


def retrieve_tickers_from_yahoo(original_url: str)->pd.DataFrame:
    """
    Scrapes Yahoo Finance for tickers and saves them to a list.
    :param original_url: string
    :return: list of strings
    """
    if '?' in original_url:
        # remove the later substring in the URL that indicates the number tickers in the total table
        original_url = original_url.split('?')[0]
    ticker_list = []
    name_list = []
    access_url = True
    counter = 0

    while access_url:
        url = original_url + '?count=100&offset={}'.format(counter * 100)
        logger.debug('Batch #{}. Accessing the following URL: {}'.format(counter + 1, url))
        source = requests.get(url=url).text
        soup = BeautifulSoup(source, 'lxml')  # lxml is the parser
        ticker_sub_list = []
        for ticker_hyper_link in soup.find_all('a', class_='Fw(600)'):
            ticker_sub_list.append(ticker_hyper_link.text)
            name_list.append(ticker_hyper_link.get('title'))
        if len(ticker_sub_list):
            logger.info("{} ticker(s) retrieved from Yahoo Finance: %s".format(len(ticker_sub_list)) % ', '.join(ticker_sub_list))
            ticker_list.extend(ticker_sub_list)
            counter += 1
            logger.debug('Sleep for 2 seconds.')
            sleep(2)
        else:
            logger.debug('Done loading tickers.')
            access_url = False
    return pd.DataFrame({'ticker': ticker_list, 'name': name_list})


def main():
    url = 'https://finance.yahoo.com/screener/unsaved/85f0e72f-6999-426b-a920-4577416f5379'
    file_name = 'china_tickers'
    tickers_df = retrieve_tickers_from_yahoo(url)  # tickers scrapped from Yahoo Finance
    save_path = __EXCEL_TICKER_FOLDER__ + '\\' + file_name + '_' + str(date.today()) + '.xlsx'
    save_df(tickers_df, full_path=save_path, sheet_name_list='tickers')
    logger.info('Tickers saved in an excel file: {}'.format(save_path))


if __name__ == '__main__':
    main()
