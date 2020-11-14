"""
strategy.py
"""
import pandas as pd
from datetime import date, datetime

import logging

# my modules
from database.financial_database import FinancialDatabase
from database.config_database import __MY_DATABASE_NAME__
from algorithmic_strategy.investment_universe import InvestmentUniverse
from dataframe_tools import merge_two_dataframes_as_of
from algorithmic_strategy import strategy_weight, strategy_signal, strategy_overlay

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
        financial_database_handler = FinancialDatabase(__MY_DATABASE_NAME__, False)
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

    def get_desc(self):
        return "{currency}, {total_return}{dividend_tax}"\
            .format(currency=self.currency.upper() if self.currency else 'local currency',
                    total_return='total return' if self.total_return else 'price return',
                    dividend_tax=' with ' + str(round(self.dividend_tax*100, 2)) + '% dividend tax' if self.dividend_tax and self.total_return else '')

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self.get_desc())


class Index(Basket):
    """Class definition of Index"""

    def __init__(self, investment_universe: InvestmentUniverse, signal=None, weight=None, overlay=None, index_fee: float = 0.0,
                 transaction_cost: float = 0.0, currency: str = None, total_return: bool = False,
                 dividend_tax: float = 0.0, observation_calendar: pd.DatetimeIndex = None, initial_value: float = 100.0,
                 strategy_weight_smoothing: int = 1, weight_observation_lag: int = 2, signal_accumulation_method: str = None):
        super().__init__(investment_universe=investment_universe, currency=currency, total_return=total_return,
                         dividend_tax=dividend_tax)
        self.signal = signal
        self.weight = weight
        self.index_fee = index_fee
        self.transaction_cost = transaction_cost
        self.initial_value = initial_value
        self.observation_calendar = observation_calendar
        self.weight_observation_lag = weight_observation_lag
        self.strategy_weight_smoothing = strategy_weight_smoothing
        self.overlay = overlay
        self._eligible_signal_accumulation_methods = ['union', 'intersection']
        self.signal_accumulation_method = signal_accumulation_method

    def get_back_test(self, end_date: datetime = None, drop_nan: bool = True) -> pd.DataFrame:
        """
        Calculate the index using the signal, weight and the weight overlays
        :param end_date: datetime
        :param drop_nan: bool
        :return: pd.DataFrame
        """
        weight_df = self.get_weight()
        price_df = self.basket_prices(start_date=weight_df.index[0], end_date=end_date)
        daily_returns = price_df.pct_change()

        # calculate the daily weights and apply smoothing
        daily_weight = merge_two_dataframes_as_of(pd.DataFrame(index=price_df.index), weight_df)
        daily_weight = daily_weight.rolling(window=self.strategy_weight_smoothing, min_periods=1).mean()

        # apply the overlay adjustments (e.g. volatility target)
        daily_returns, daily_weight = self._apply_weight_modifier(daily_return_df=daily_returns, daily_weight_df=daily_weight)

        # TODO need to check if the weight obs lag is ok

        index_result_df = index_calculation(daily_returns=daily_returns, daily_weight=daily_weight,
                                            weight_observation_lag=self.weight_observation_lag,
                                            transaction_cost=self.transaction_cost, index_fee=self.index_fee,
                                            initial_value=self.initial_value)
        # clean results when applicable
        if drop_nan:
            index_result_df.dropna(inplace=True)
        return index_result_df

    def _apply_weight_modifier(self, daily_return_df: pd.DataFrame, daily_weight_df: pd.DataFrame) -> tuple:
        """
        The daily return and weight DataFrames are adjusted based upon a specified 'overlay' e.g. volatility target
        mechanism or beta hedge. In case a list of overlays have been specified, they are applied one after the other
        :param daily_return_df: pd.DataFrame
        :param daily_weight_df: pd.DataFrame
        :return: pd.DataFrame, pd.DataFrame
        """
        if self.overlay is None:
            logger.debug('no overlay is specified')
            return daily_return_df, daily_weight_df
        elif isinstance(self.overlay, list):
            logger.debug('applying a list of {} overlays to the daily returns and weights'.format(len(self.overlay)))
            # apply each weight overlays
            for weight_overlay in self.overlay:
                daily_return_df, daily_weight_df = weight_overlay.get_return_weight_tuple_after_scaling(
                    multivariate_daily_return_df=daily_return_df,
                    multivariate_daily_weight_df=daily_weight_df)
        else:
            logger.debug('applying an overlay to the daily returns and weights')
            daily_return_df, daily_weight_df = self.overlay.get_return_weight_tuple_after_scaling(multivariate_daily_return_df=daily_return_df, multivariate_daily_weight_df=daily_weight_df)
        return daily_return_df, daily_weight_df

    def get_index_desc_df(self):
        """
        Creates a DataFrame where all rows represents descriptions of the index components
        :return: pd.DataFrame
        """
        index_desc_df = pd.DataFrame(columns=['Index description'])
        index_desc_df.loc['Index eligibility:'] = self.investment_universe.get_desc()
        index_desc_df.loc['Currency:'] = self.currency.upper() if self.currency else 'Local currency'
        index_desc_df.loc['Return type:'] = 'Total return' if self.total_return else 'Price return'
        if self.dividend_tax:
            index_desc_df.loc['Dividend tax:'] = str(round(100 * self.dividend_tax, 2)) + '%'
        if self.signal is not None:
            if isinstance(self.signal, list):
                index_desc_df.loc['Signal accumulation method:'] = self.signal_accumulation_method + ' of all signals'
                signal_counter = 1
                for signal in self.signal:
                    index_desc_df.loc['Signal ({}/{}):'.format(signal_counter, len(self.signal))] = type(signal).__name__ + ' (' + signal.get_desc() + ')'
                    signal_counter += 1
            else:
                index_desc_df.loc['Signal:'] = type(self.signal).__name__ + ' (' + self.signal.get_desc() + ')'
        # if self.signal is not None:
        #     index_desc_df.loc['Signal:'] = type(self.signal).__name__ + ' (' + self.signal.get_desc() + ')'
        index_desc_df.loc['Weight:'] = type(self.weight).__name__ + ' (' + self.weight.get_weight_desc() + ')'
        if self.overlay is not None:
            if isinstance(self.overlay, list):
                overlay_counter = 1
                for overlay in self.overlay:
                    index_desc_df.loc['Overlay ({}/{}):'.format(overlay_counter, len(self.overlay))] = type(overlay).__name__ + ' (' + overlay.get_desc() + ')'
                    overlay_counter += 1
            else:
                index_desc_df.loc['Overlay:'] = type(self.overlay).__name__ + ' (' + self.overlay.get_desc() + ')'
        index_desc_df.loc['Index fee p.a.:'] = str(round(100 * self.index_fee, 2)) + '%'
        index_desc_df.loc['Transaction costs:'] = str(round(100 * self.transaction_cost, 2)) + '%'
        if self.strategy_weight_smoothing > 1:
            index_desc_df.loc['Weight smoothing (days):'] = str(self.strategy_weight_smoothing)
        index_desc_df.loc['Weight observation lag (days):'] = str(self.weight_observation_lag)
        return index_desc_df

    def _get_eligibility_df(self):
        eligibility_df = self.investment_universe.get_eligibility_df(True)
        if self.observation_calendar is not None:
            eligibility_df = merge_two_dataframes_as_of(pd.DataFrame(index=self.observation_calendar), eligibility_df)
        return eligibility_df

    def get_weight(self):
        self.weight.signal_df = self.get_signal()
        return self.weight.get_weights()

    def get_signal(self):
        eligibility_df = self._get_eligibility_df()
        if self.signal is None:
            logger.info('since no signal has been specified, the eligibility DataFrame from the investment universe '
                        'is used')
            return strategy_signal.Signal(eligibility_df=eligibility_df).get_signal()
        elif isinstance(self.signal, list):
            # calculate the accumulative signal
            return self._get_accumulated_signal(eligibility_df)
        else:
            self.signal.eligibility_df = eligibility_df
            return self.signal.get_signal()

    def _get_accumulated_signal(self, eligibility_df: pd.DataFrame):
        """
        Depending on the class attribute signal_accumulation_method return a signal DataFrame that is either the union
        or intersection
        :param eligibility_df:
        :return: pd.DataFrame
        """
        logger.info('calculate the {} of {} signals'.format(self.signal_accumulation_method.lower(),
                                                            len(self.signal)))
        # check the aggregator method
        accumulated_signal = None
        if self.signal_accumulation_method is None:
            raise ValueError('signal_accumulation_method is not specified')
        elif self.signal_accumulation_method.lower() == self._eligible_signal_accumulation_methods[0]:
            # take the union of the signals by looking at the maximum signal value
            for signal in self.signal:
                signal.eligibility_df = eligibility_df
                if accumulated_signal is None:
                    accumulated_signal = signal.get_signal()
                else:
                    # calculate the element-wise maximum of the signal values
                    accumulated_signal = pd.concat([accumulated_signal, signal.get_signal()]).max(level=0)
        elif self.signal_accumulation_method.lower() == self._eligible_signal_accumulation_methods[1]:
            # take the intersection/product of the signal by updating the eligibility DataFrame for each signal
            for signal in self.signal:
                signal.eligibility_df = eligibility_df
                if accumulated_signal is None:
                    accumulated_signal = signal.get_signal()
                else:
                    # calculate the element-wise product of the signal values
                    accumulated_signal *= signal.get_signal().values
        return accumulated_signal

    def get_desc(self):
        return super().get_desc()

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        if signal is None:
            self._signal = None
        elif issubclass(type(signal), strategy_signal.Signal) or isinstance(signal, strategy_signal.Signal) \
                or isinstance(signal, list):
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
        elif issubclass(type(weight), strategy_weight.Weight):
            self._weight = weight
        else:
            raise ValueError('Needs to be of type that is a subclass of Weight.')

    @property
    def overlay(self):
        return self._weight_modifier

    @overlay.setter
    def overlay(self, weight_modifier):
        if weight_modifier is None:
            self._weight_modifier = None
        elif issubclass(type(weight_modifier), strategy_overlay._Overlay):
            self._weight_modifier = weight_modifier
        elif isinstance(weight_modifier, list):
            if all(issubclass(type(w_mod), strategy_overlay._Overlay) for w_mod in weight_modifier):
                self._weight_modifier = weight_modifier
            else:
                raise ValueError('All elements in the weight_modifier list is not a subclass of _WeightAdjustment')
        else:
            raise ValueError('weight_modifier needs to be an object or a list of objects that are a subclass of _WeightAdjustment.')

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
    def strategy_weight_smoothing(self):
        return self._strategy_weight_smoothing

    @strategy_weight_smoothing.setter
    def strategy_weight_smoothing(self, strategy_weight_smoothing: int):
        if strategy_weight_smoothing < 1:
            raise ValueError('strategy_weight_smoothing needs to be an int larger or equal to 1.')
        self._strategy_weight_smoothing = strategy_weight_smoothing

    @property
    def signal_accumulation_method(self):
        return self._signal_accumulation_method

    @signal_accumulation_method.setter
    def signal_accumulation_method(self, signal_accumulation_method: str):
        if signal_accumulation_method is None:
            self._signal_accumulation_method = None
        elif signal_accumulation_method.lower() in self._eligible_signal_accumulation_methods:
            self._signal_accumulation_method = signal_accumulation_method.lower()
        else:
            raise ValueError("signal_accumulation_method needs to be equal to '%s'" % "' or '".join(self._eligible_signal_accumulation_methods))


def index_calculation(daily_returns: pd.DataFrame, daily_weight: pd.DataFrame, weight_observation_lag: int,
                      transaction_cost: float, index_fee: float, initial_value: float):
    """
    Calculates an index as the rolling product sum of the daily returns and weights. Transaction costs are taken on on
    the absolute value of daily changes in the weights and the index fee p.a. is subtracted from the final result
    :param daily_returns: pd.DataFrame
    :param daily_weight: pd.DataFrame
    :param weight_observation_lag: int
    :param transaction_cost: float
    :param index_fee: float (p.a.)
    :param initial_value: float
    :return: pd.DataFrame
    """

    # clean the daily returns
    daily_returns.fillna(0, inplace=True)

    # calculate the gross return of the index
    col_name = 'gross_index'
    gross_instrument_returns = daily_returns * daily_weight.shift(weight_observation_lag).values
    index_return_df = pd.DataFrame({'gross_return': gross_instrument_returns.sum(axis=1, skipna=False).values},
                                   index=gross_instrument_returns.index)

    # if applicable, calculate the return net of transaction costs and fees
    if transaction_cost > 0 or index_fee != 0:
        # net of transaction costs
        abs_weight_delta = daily_weight.diff().abs().sum(axis=1)
        transaction_cost_df = pd.DataFrame({'TC': transaction_cost * abs_weight_delta.values},
                                           index=abs_weight_delta.index)
        index_return_df -= transaction_cost_df.values

        # net of index fees
        dt = [None] + [(index_return_df.index[n] - index_return_df.index[n - 1]).days / 365 for n in
                       range(1, len(index_return_df.index))]
        dt_s = pd.Series(data=dt, index=index_return_df.index)
        index_fee_df = pd.DataFrame({'FEE': index_fee * dt_s.values}, index=dt_s.index)
        index_return_df -= index_fee_df.values
        index_return_df.columns = ['net_return']
        col_name = 'net_index'

    # calculate the index
    index_result_df = index_return_df.copy()
    start_of_index_i = index_return_df.index.get_loc(index_return_df.first_valid_index()) - 1
    index_result_df = (1 + index_result_df).cumprod()  # (1 + r_1) * (1 + r_2) * ...
    index_result_df.iloc[start_of_index_i, :] = 1.0  # start at 1.0
    index_result_df *= initial_value  # scale the index by the initial value
    index_result_df.columns = [col_name]
    return index_result_df




