"""
index.py
"""
import pandas as pd
from datetime import date, datetime

import logging

# my modules
from financial_database import FinancialDatabase
from config_database import my_database_name
from investment_universe import InvestmentUniverse
import index_signal
import index_weight_new

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Basket:
    """Class definition of Basket"""

    def __init__(self, investment_universe: InvestmentUniverse, currency: str = None, total_return: bool = False, dividend_tax: float = 0.0):
        self.investment_universe = investment_universe
        self.currency = currency
        self.total_return = total_return
        self.dividend_tax = dividend_tax

    def basket_prices(self, start_date: {date, datetime}=None, end_date: {date, datetime}=None):
        logger.debug('Get basket price.')
        financial_database_handler = FinancialDatabase(my_database_name, False)
        tickers = self.investment_universe.get_eligible_tickers()
        if self.total_return:
            return financial_database_handler.get_total_return_df(tickers, start_date, end_date, self.dividend_tax,
                                                                  self.currency)
        else:
            return financial_database_handler.get_close_price_df(tickers, start_date, end_date, self.currency)

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
            .format(len(self.investment_universe.get_eligible_tickers()), currency=self.currency if self.currency else 'local',
                    total_return='total return' if self.total_return else 'price return',
                    dividend_tax=' with ' + str(round(self.dividend_tax*100, 2)) + '% dividend tax' if self.dividend_tax and self.total_return else '')


class Index(Basket):
    """Class definition of Index"""

    def __init__(self, investment_universe: InvestmentUniverse, signal=None, weight=None, index_fee: float = 0.0,
                 transaction_cost: float = 0.0, currency: str = None, total_return: bool = False,
                 dividend_tax: float = 0.0, observation_calendar: pd.DatetimeIndex = None, initial_value: float = 100.0,
                 daily_rebalancing: bool = True):
        super().__init__(investment_universe=investment_universe, currency=currency, total_return=total_return,
                         dividend_tax=dividend_tax)
        self.signal = signal
        self.weight = weight
        self.index_fee = index_fee
        self.transaction_cost = transaction_cost
        self.initial_value = initial_value
        self.observation_calendar = observation_calendar
        self.daily_rebalancing = daily_rebalancing

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        if issubclass(type(signal), index_signal._Signal):
            self._signal = signal
        else:
            raise ValueError('Needs to be of type that is a subclass of _Signal.')

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        if issubclass(type(weight), index_weight_new.Weight):
            self._weight = weight
        else:
            raise ValueError('Needs to be of type that is a subclass of Weight.')

    @property
    def index_fee(self):
        return self._index_fee

    @index_fee.setter
    def index_fee(self, index_fee: float):
        if index_fee < 0:
            logger.warning('index_fee is negative.')
        self._index_fee = index_fee

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
    def observation_calendar(self):
        return self._observation_calendar

    @observation_calendar.setter
    def observation_calendar(self, observation_calendar: pd.DatetimeIndex):
        """
        Check if the observation calendar is monotonically increasing.
        :param observation_calendar: DatetimeIndex
        :return: None
        """
        if observation_calendar is None:
            self._observation_calendar = None
        elif observation_calendar.is_monotonic_increasing and isinstance(observation_calendar, pd.DatetimeIndex):
            self._observation_calendar = observation_calendar
        else:
            ValueError('observation_calendar needs to be a DatetimeIndex that is monotonic increasing.')


