from bs4 import BeautifulSoup
import requests
from time import sleep
import logging

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


def retrieve_tickers_from_yahoo(original_url: str):
    ticker_list = []
    retrieve_tickers = True
    i = 0
    while retrieve_tickers:
        url = original_url + '?count=100&offset={}'.format(i * 100)
        logger.debug('Accessing the following URL: {}'.format(url))
        source = requests.get(url=url).text
        logger.debug('Parse the HTML using BeautifulSoup.')
        soup = BeautifulSoup(source, 'lxml')  # lxml is the parser
        ticker_sub_list = []
        for ticker_area_1 in soup.find_all('td', class_='Va(m) Ta(start) Pstart(6px) Pend(10px) Miw(90px) Start(0) Pend(10px) simpTblRow:h_Bgc($extraLightBlue) Bgc(white) Ta(start)! Fz(s)'):
            ticker_sub_list.append(ticker_area_1.a.text)
        for ticker_area_2 in soup.find_all('td', class_='Va(m) Ta(start) Pstart(6px) Pend(10px) Miw(90px) Start(0) Pend(10px) simpTblRow:h_Bgc($extraLightBlue) Bgc($altRowColor) Ta(start)! Fz(s)'):
            ticker_sub_list.append(ticker_area_2.a.text)
        if len(ticker_sub_list):
            logger.info("{} ticker(s) retrieved from Yahoo Finance: %s".format(len(ticker_sub_list)) % ', '.join(ticker_sub_list))
            ticker_list.extend(ticker_sub_list)
            logger.debug('Sleep for 2 seconds.')
            i += 1
            sleep(2)
        else:
            logger.debug('Done loading tickers.')
            retrieve_tickers = False
    ticker_list.sort()
    return ticker_list


