import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import date, datetime
import logging
import matplotlib.pyplot as plt

# my own modules
from financial_database import FinancialDatabase
from index_signal import Signal
from index_weight import _Weight, EqualWeight
from finance_tools import index_calculation
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar

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
                 dividend_tax: float = 0.0, initial_amount: float = 100.0):
        super().__init__(tickers, currency, total_return, dividend_tax)
        self.rebalancing_calendar = rebalancing_calendar
        self.index_fee = index_fee
        self.transaction_cost = transaction_cost
        self.initial_amount = initial_amount
        self._signal = None
        self._weight = None

    def _check_before_back_test(self):
        if self.signal is None:
            self.signal = Signal(ticker_list=self.tickers)  # default signal
        if self.weight is None:
            raise ValueError('No weight assigned.')

    def get_back_test(self, end_date: {date, datetime}=None, return_index_only: bool = True):
        back_test = self._get_back_test_or_weight_df(True, end_date)
        if return_index_only:
            return back_test[['index']]
        return back_test

    def get_weight_df(self, end_date: {date, datetime}=None):
        return self._get_back_test_or_weight_df(True, end_date)

    def _get_back_test_or_weight_df(self, get_back_test: bool, end_date: {date, datetime}=None):
        # handle the start and end date
        start_date = self.rebalancing_calendar[0] - BDay(5)
        if end_date is not None and np.datetime64(end_date) <= np.datetime64(self.rebalancing_calendar[0]):
            raise ValueError("end_date is not allowed to be before the rebalancing calendar.")

        # retrieve the underlying price to be used in the index
        underlying_price_df = self.basket_prices(start_date, end_date)
        self._check_before_back_test()

        # adjust rebalance calendar by moving one business day ahead in the underlying price calendar

        # calculate the signal and if there is no observation calendar assigned to the signal assign a default one
        if self.signal.signal_observation_calendar is None:
            self.signal.signal_observation_calendar = underlying_price_df.index
        signal_df = self.signal.get_signal_df()

        # calculate the weights
        self.weight.signal_df = signal_df
        weight_df = self.weight.get_weights()
        weight_df = select_rows_from_dataframe_based_on_sub_calendar(weight_df, self.rebalancing_calendar)

        if get_back_test:
            return index_calculation(underlying_price_df, weight_df, self.transaction_cost, self.index_fee,
                                     self.initial_amount)
        else:
            return weight_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        if issubclass(type(signal), Signal) or isinstance(signal, Signal):
            self._signal = signal
        else:
            raise ValueError('Needs to be an object from Signal class or a subclass of class Signal.')

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        if issubclass(type(weight), _Weight):
            self._weight = weight
        else:
            raise ValueError('Needs to be an object from a subclass of class _Weight.')

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

    @property
    def initial_amount(self):
        return self._initial_amount

    @initial_amount.setter
    def initial_amount(self, initial_amount: float):
        if initial_amount > 0:
            self._initial_amount = initial_amount
        else:
            raise ValueError('initial_amount needs to be greater than zero.')


def main():
    tickers = ["SAND.ST", "HM-B.ST", "AAK.ST"]
    rebalance_cal = pd.date_range(start='2010', periods=10, freq='M')
    index_main = Index(tickers, rebalance_cal, total_return=True, dividend_tax=0.01, index_fee=-0.03)
    index_main.weight = EqualWeight()
    end_date = date(2010, 6, 3)
    back_test = index_main.get_back_test(end_date=end_date)
    print(back_test)

    from financial_database import FinancialDatabase
    from config_database import my_database_name
    print(FinancialDatabase(my_database_name, False).get_close_price_df(tickers, rebalance_cal[0], end_date))
    # back_test.plot()
    # plt.show()


def main_dates():
    rebalance_cal = pd.date_range(start='2010', periods=10, freq='M')
    cal = pd.date_range(start='2010', end=rebalance_cal[0], freq='D')

    rebalance_cal_list = list(rebalance_cal)
    cal_list = list(cal)
    rebalance_cal_list_adj = []
    for reb_date in rebalance_cal_list:
        if reb_date in cal_list:
            reb_date_adj = reb_date
        else:
            date_diff = [date_in_cal - reb_date for date_in_cal in cal if date_in_cal - reb_date > 0]
            print(date_diff)

if __name__ == '__main__':
    main_dates()

