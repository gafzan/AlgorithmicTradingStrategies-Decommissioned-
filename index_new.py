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
from dataframe_tools import merge_two_dataframes_as_of
from finance_tools import index_daily_rebalanced
import index_signal_new
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

    def basket_prices(self, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                      forward_fill_na: bool = True):
        logger.debug('Get basket price.')
        financial_database_handler = FinancialDatabase(my_database_name, False)
        tickers = self.investment_universe.get_eligible_tickers()
        if self.total_return:
            price = financial_database_handler.get_total_return_df(tickers, start_date, end_date, self.dividend_tax,
                                                                  self.currency)
        else:
            price = financial_database_handler.get_close_price_df(tickers, start_date, end_date, self.currency)
        if forward_fill_na:
            price.fillna(inplace=True, method='ffill')
        return price

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
                 daily_rebalancing: bool = True, weight_rebalance_lag: int = 1, weight_observation_lag: int = 2,
                 volatility_target: float = None, volatility_observation_lag: {int, list}=None, risky_weight_cap: float = 1.0):
        super().__init__(investment_universe=investment_universe, currency=currency, total_return=total_return,
                         dividend_tax=dividend_tax)
        self.signal = signal
        self.weight = weight
        self.index_fee = index_fee
        self.transaction_cost = transaction_cost
        self.initial_value = initial_value
        self.observation_calendar = observation_calendar
        self.daily_rebalancing = daily_rebalancing
        self.weight_observation_lag = weight_observation_lag
        self.weight_rebalance_lag = weight_rebalance_lag
        self.volatility_target = volatility_target
        self.volatility_observation_lag = volatility_observation_lag
        self.risky_weight_cap = risky_weight_cap

    def get_back_test(self, end_date: datetime = None, only_index: bool = False):
        self._assign_signal_to_weight()
        weight_df = self.weight.get_weights()
        price_df = self.basket_prices(start_date=weight_df.index[0], end_date=end_date)
        daily_returns = price_df.pct_change()
        daily_returns.iloc[0, :] = 0
        index_result = index_daily_rebalanced(daily_returns, weight_df, self.transaction_cost, self.index_fee,
                                              self.weight_rebalance_lag, self.weight_observation_lag, self.initial_value,
                                              self.volatility_target, self.volatility_observation_lag, self.risky_weight_cap)
        if only_index:
            try:
                return index_result.loc[:, ['NET_INDEX']].dropna()
            except KeyError:
                return index_result.loc[:, ['GROSS_INDEX']].dropna()
        else:
            return index_result

    def _get_eligibility_df(self):
        eligibility_df = self.investment_universe.get_eligibility_df(True)
        if self.observation_calendar is not None:
            eligibility_df = merge_two_dataframes_as_of(pd.DataFrame(index=self.observation_calendar), eligibility_df)
        return eligibility_df

    def _assign_signal_to_weight(self):
        eligibility_df = self._get_eligibility_df()
        if self.signal is None:
            signal = index_signal_new.Signal(eligibility_df=eligibility_df)  # default signal is just the eligibility DataFrame
            self.weight.signal_df = signal.get_signal()
        else:
            self.signal.eligibility_df = eligibility_df
            self.weight.signal_df = self.signal.get_signal()

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        if signal is None:
            self._signal = None
        elif issubclass(type(signal), index_signal_new.Signal) or isinstance(signal, index_signal_new.Signal):
            self._signal = signal
        else:
            raise ValueError('Needs to be of type that is a subclass of _Signal.')

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        if weight is None:
            self._weight = None
        elif issubclass(type(weight), index_weight_new.Weight):
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

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value: float):
        if initial_value <= 0:
            raise ValueError('initial_value needs to be a float larger than 0.')
        self._initial_value = initial_value

    @property
    def weight_observation_lag(self):
        return self._weight_observation_lag

    @weight_observation_lag.setter
    def weight_observation_lag(self, weight_observation_lag: int):
        if weight_observation_lag < 1:
            raise ValueError('weight_observation_lag needs to be an int larger or equal to 1 in order for the index to '
                             'be replicable.')
        self._weight_observation_lag = weight_observation_lag

    @property
    def weight_rebalance_lag(self):
        return self._weight_rebalance_lag

    @weight_rebalance_lag.setter
    def weight_rebalance_lag(self, weight_rebalance_lag: int):
        if weight_rebalance_lag < 1:
            raise ValueError('weight_rebalance_lag needs to be an int larger or equal to 1.')
        self._weight_rebalance_lag = weight_rebalance_lag

    @property
    def volatility_target(self):
        return self._volatility_target

    @volatility_target.setter
    def volatility_target(self, volatility_target: float):
        if volatility_target is None:
            self._volatility_target = None
        elif volatility_target <= 0:
            raise ValueError('volatility_target needs to be a float larger than 0.')
        self._volatility_target = volatility_target

    @property
    def risky_weight_cap(self):
        return self._risky_weight_cap

    @risky_weight_cap.setter
    def risky_weight_cap(self, risky_weight_cap: float):
        if risky_weight_cap <= 0:
            raise ValueError('risky_weight_cap needs to be a float larger than 0.')
        self._risky_weight_cap = risky_weight_cap

    def __repr__(self):
        return '<Index()>'


def main():
    # investment universe
    tickers = ["AGRO.ST", "AAK.ST", "ABB.ST"]
    invest_uni = InvestmentUniverse(tickers, '2018', '2020', freq='M')
    invest_uni.apply_liquidity_filter(60, 300000, 'sek')

    index = Index(invest_uni, transaction_cost=0.001, index_fee=0.005, volatility_target=0.1,
                  volatility_observation_lag=60)
    index.weight = index_weight_new.EqualWeight()
    eqw_index = index.get_back_test(only_index=True)
    index.weight = index_weight_new.VolatilityWeight(60)
    inv_vol_index = index.get_back_test(only_index=True)
    combined = eqw_index.join(inv_vol_index, rsuffix='_inverse_volatility'.upper())


    from matplotlib import pyplot as plt
    print(combined)
    combined.plot()
    plt.show()


if __name__ == '__main__':
    main()
