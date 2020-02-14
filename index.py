import pandas as pd
from datetime import date, datetime
import logging

# my own modules
from financial_database import FinancialDatabase
from index_signal import Signal

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Basket:
    """Class definition of Basket"""

    database_name = r'sqlite:///C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\stock_db_v3'

    def __init__(self, tickers: {str, list}, currency: str = None, total_return: bool = False, dividend_tax: float = 0.0):
        self.tickers = tickers
        self.currency = currency
        self.total_return = total_return
        self.dividend_tax = dividend_tax

    def basket_prices(self, start_date: {date, datetime}=None, end_date: {date, datetime}=None):
        financial_database_handler = FinancialDatabase(self.database_name, False)
        if self.total_return:
            return financial_database_handler.get_close_price_df(self.tickers, start_date, end_date, self.currency)
        else:
            return financial_database_handler.get_total_return_df(self.tickers, start_date, end_date, self.dividend_tax,
                                                                  self.currency)

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods

    @property
    def dividend_tax(self):
        return self._dividend_tax

    @dividend_tax.setter
    def dividend_tax(self, dividend_tax: float):
        if dividend_tax >= 0:
            self._dividend_tax = dividend_tax
        else:
            raise ValueError('dividend_tax needs to be greater or equal to zero.')

    def __repr__(self):
        return "<Basket(#tickers = {}, currency = {currency}, {total_return}{dividend_tax})>"\
            .format(len(self.tickers), currency=self.currency if self.currency else 'local',
                    total_return='total return' if self.total_return else 'price return',
                    dividend_tax=' with ' + str(round(self.dividend_tax*100, 2)) + '% dividend tax' if self.dividend_tax and self.total_return else '')


class Index(Basket):
    """Class definition of Index. Subclass of Basket class."""

    def __init__(self, tickers: {str, list}, rebalancing_calendar: pd.DatetimeIndex, index_fee: float = 0.0,
                 transaction_cost: float = 0.0, currency: str = None, total_return: bool = False,
                 dividend_tax: float = 0.0):
        super().__init__(tickers, currency, total_return, dividend_tax)
        self.rebalancing_calendar = rebalancing_calendar
        self.index_fee = index_fee
        self.transaction_cost = transaction_cost
        self._signal = None
        self._weight = None

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        if issubclass(signal, Signal) or isinstance(signal, Signal):
            self._signal = signal
        else:
            raise ValueError('Needs to be an object from Signal class or a subclass of the Signal class')

    @property
    def rebalancing_calendar(self):
        return self._rebalancing_calendar

    @rebalancing_calendar.setter
    def rebalancing_calendar(self, rebalancing_calendar: pd.DatetimeIndex):
        if rebalancing_calendar.is_monotonic_increasing:
            self._rebalancing_calendar = rebalancing_calendar
        else:
            raise ValueError('rebalancing_calendar needs to be monotonic increasing (oldest to newest date).')

    @property
    def index_fee(self):
        return self._index_fee

    @index_fee.setter
    def index_fee(self, index_fee: float):
        self._index_fee = index_fee
        if index_fee < 0:
            logger.warning('index_fee is negative.')

    @property
    def transaction_cost(self):
        return self._transaction_cost

    @transaction_cost.setter
    def transaction_cost(self, transaction_cost: float):
        if transaction_cost >= 0:
            self._transaction_cost = transaction_cost
        else:
            raise ValueError('transaction_cost needs to be greater or equal to zero.')


def main():
    tickers = ["SAND.ST", "HM-B.ST", "AAK.ST"]
    rebalance_cal = pd.date_range(start='2010', periods=5, freq='M')
    index_main = Index(tickers, rebalance_cal, total_return=True, dividend_tax=0.01, index_fee=-0.03)
    print(index_main.signal)



if __name__ == '__main__':
    main()

