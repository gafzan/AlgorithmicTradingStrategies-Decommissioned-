"""
strategy_weight.py
"""
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas.core.common as com
import numpy as np

import logging

# my modules
from database.financial_database import FinancialDatabase
from database.config_database import my_database_name
from dataframe_tools import merge_two_dataframes_as_of
from financial_analysis.finance_tools import realized_volatility


# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Weight:
    """Class definition of Weight"""

    # Initializer / instance attribute
    def __init__(self, signal_df: pd.DataFrame = None, max_instrument_weight: float = None,
                 min_instrument_weight: float = None):
        self._signal_df = signal_df
        self.max_instrument_weight = max_instrument_weight
        self.min_instrument_weight = min_instrument_weight

    def get_weights(self):
        if self.signal_df is None:
            raise ValueError('signal_df not yet assigned.')
        weight_df = self._calculate_weight()
        if com.count_not_none(self.max_instrument_weight, self.min_instrument_weight) > 0:
            weight_df = self._apply_basic_constraint(weight_df)
        return weight_df

    def _apply_basic_constraint(self, weight_df):
        """
        Applies min max constraint on the weights in weight_df. This is the most basic form of constraint.
        :param weight_df: pd.DataFrame
        :return: pd.DataFrame
        """

        max_instrument_w = 999.0 if self.max_instrument_weight is None else self.max_instrument_weight
        min_instrument_w = -999.0 if self.min_instrument_weight is None else self.min_instrument_weight
        weight_df = weight_df.copy()
        for col_name in list(weight_df):
            # apply the lambda function that adds the constraints
            weight_df.loc[:, col_name] = weight_df[col_name].apply(
                lambda x: max(min_instrument_w, min(max_instrument_w, x))
                if not np.isnan(x)
                else np.nan
            )
        return weight_df

    def _calculate_weight(self):
        # to be overridden
        raise ValueError('Only an instance of a subclass of Weight object should call calculate_weight.')

    @property
    def signal_df(self):
        return self._signal_df

    @signal_df.setter
    def signal_df(self, signal_df: pd.DataFrame):
        if signal_df.index.is_monotonic_increasing and isinstance(signal_df.index, pd.DatetimeIndex):
            self._signal_df = signal_df
        else:
            raise ValueError('Index of signal_df needs to be a monotonically increasing DatetimeIndex.')

    @staticmethod
    def get_weight_desc():
        return ''

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self.get_weight_desc())


class _ProportionalWeight(Weight):
    """Class definition of _ProportionalWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame, inversely: bool):
        super().__init__(signal_df)
        self.inversely = inversely

    def _get_dataframe(self):
        raise ValueError('_get_dataframe should not be called by an instance of _ProportionalWeight.')

    def _calculate_weight(self):
        data = self._get_dataframe()
        if list(data) == list(self.signal_df):  # same column names
            if self.inversely:
                data = data.apply(lambda x: 1 / x)  # inverse of all values
            data_obs_dates_df = merge_two_dataframes_as_of(pd.DataFrame(index=self.signal_df.index), data)
            data.replace([np.nan, np.Inf, -np.Inf], 0, inplace=True)
            eligible_data_indicator = self.signal_df.copy()
            eligible_data_indicator[~eligible_data_indicator.isnull()] = 1  # set all non NaN to 1
            data_obs_dates_df *= eligible_data_indicator.values

            data_sum = data_obs_dates_df.sum(axis=1)
            weight_df = data_obs_dates_df.divide(data_sum, axis=0)

            # make sure that the weights have the same observation calendar as the signal DataFrame
            weight_df *= self.signal_df.values

            weight_df.replace(np.nan, 0, inplace=True)
            return weight_df
        else:
            raise ValueError('column headers of signal_df and the given data are not the same.')


class _FinancialDatabaseDependentWeight(Weight):
    """Class definition of _FinancialDatabaseDependentWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame):
        Weight.__init__(self, signal_df=signal_df)
        self._financial_database_handler = FinancialDatabase(my_database_name)

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def financial_database_handler(self):
        # make financial_database_handler read-only
        return self._financial_database_handler


class _PriceBasedWeight(_FinancialDatabaseDependentWeight):
    """Class definition of _PriceBasedSignal. Subclass of _FinancialDatabaseDependentWeight."""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int}):
        _FinancialDatabaseDependentWeight.__init__(self, signal_df=signal_df)
        self.total_return = total_return
        self.currency = currency
        self._observation_buffer = 0
        self._weekday_i_dict = {'mon': 0, 'monday': 0, 'tuesday': 1, 'tue': 1, 'wed': 2, 'wednesday': 2, 'thursday': 3,
                                'thu': 4, 'friday': 5, 'fri': 5}
        self.price_obs_freq = price_obs_freq

    def _get_start_end_date(self):
        if self.signal_df is None:
            raise ValueError('signal_df needs to be specified.')
        return min(self.signal_df.index) - BDay(self._observation_buffer), max(self.signal_df.index)

    def _get_price_df(self):
        start_date, end_date = self._get_start_end_date()
        if self.total_return:
            price = self.financial_database_handler.get_total_return_df(list(self.signal_df), start_date, end_date,
                                                                        0, self.currency)
        else:
            price = self.financial_database_handler.get_close_price_df(list(self.signal_df), start_date, end_date,
                                                                       self.currency)
        # filter out rows if you have specified certain observation intervals or weekdays
        if isinstance(self.price_obs_freq, str):
            price = price[price.index.weekday == self._weekday_i_dict[self.price_obs_freq]]
        elif isinstance(self.price_obs_freq, int):
            # sort index in descending order. this is done to have the count start from the latest observation date
            price = price.sort_index(ascending=False).iloc[::self.price_obs_freq, :].sort_index()
        return price

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def price_obs_freq(self):
        return self._price_obs_freq

    @price_obs_freq.setter
    def price_obs_freq(self, price_obs_freq: {str, int}):
        if isinstance(price_obs_freq, str) and price_obs_freq.lower() in list(self._weekday_i_dict.keys()):
            self._price_obs_freq = price_obs_freq.lower()
        elif isinstance(price_obs_freq, int) and price_obs_freq >= 1:
            self._price_obs_freq = price_obs_freq
        elif price_obs_freq is None:
            self._price_obs_freq = None
        else:
            raise ValueError('price_obs_freq needs an int larger or equal to 1 or a string equal to %s.'
                             % ' or '.join(self._weekday_i_dict.keys()))


class _PriceBasedProportionalWeight(_PriceBasedWeight, _ProportionalWeight):
    """Class definition of _PriceBasedProportionalWeight. Subclass of _PriceBasedWeight and
    _ProportionalWeight."""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int},
                 inversely: bool):
        _PriceBasedWeight.__init__(self, signal_df=signal_df, total_return=total_return, currency=currency,
                                   price_obs_freq=price_obs_freq)
        _ProportionalWeight.__init__(self, signal_df=signal_df, inversely=inversely)


class EqualWeight(Weight):
    """Class definition of EqualWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame = None, gross_equal_weights: bool = True):
        """
        If gross_equal_weights is True the weights are calculated as equal weights regardless if it is a long or short
        position. If False, the equal weighting is separately done for the long and short positions.
        :param signal_df: DataFrame
        :param gross_equal_weights: bool
        """
        super().__init__(signal_df=signal_df)
        self.gross_equal_weights = gross_equal_weights

    def _calculate_weight(self):
        # calculate the number of non-NaN columns per row
        if self.gross_equal_weights:
            numeric_columns = self.signal_df.count(axis=1)
            weight_df = self.signal_df.divide(numeric_columns, axis=0).replace(np.nan, 0)
        else:
            positive_columns = self.signal_df[self.signal_df > 0.0].count(axis=1)
            negative_columns = self.signal_df[self.signal_df < 0.0].count(axis=1)
            weight_df = self.signal_df.copy().replace(np.nan, 0)
            weight_df[weight_df > 0.0] = weight_df[weight_df > 0.0].divide(positive_columns, axis=0)
            weight_df[weight_df < 0.0] = weight_df[weight_df < 0.0].divide(negative_columns, axis=0)
        return weight_df


class StaticWeight(Weight):
    """Class definition of StaticWeight. Subclass of Weight."""

    def _calculate_weight(self):
        """
        Ask user to input a value for the weight for each ticker and signal value
        :return: DataFrame
        """
        counter = 0
        weight_df = pd.DataFrame(index=self.signal_df.index, columns=self.signal_df.columns)
        for ticker in list(self.signal_df):
            signal_values_for_ticker = set(self.signal_df.loc[:, ticker].values)  # unique signal values
            # as a feature nan != nan. Use this to filter out the nan
            signal_values_for_ticker = {signal for signal in signal_values_for_ticker if signal == signal}
            # ask user to insert each weight corresponding to the values of the signal (except 0) for each ticker
            signal_weight_dict = {}
            for signal_value in signal_values_for_ticker:
                msg = 'Enter weight(%) for {} (#{} of {} tickers) when signal = {}: '.format(ticker, counter + 1, self.signal_df.shape[0], signal_value)
                weight_from_user = self.ask_user_for_weight(msg)
                signal_weight_dict.update({signal_value: weight_from_user / 100})
            signal_weight_dict.update({np.nan: 0.0})
            weight_df[ticker] = self.signal_df[ticker].map(signal_weight_dict)
            counter += 1
        # check the values and log a warning if total allocation is different from 100%
        total_weights = pd.DataFrame(index=weight_df.index, data=weight_df.sum(axis=1).values, columns=['total_weight'])
        obs_dates_with_distinctive_allocation = total_weights[total_weights['total_weight'] != 1.0].index
        if len(obs_dates_with_distinctive_allocation):
            logger.warning('There are {} observation dates where the total allocation is different from 100%.'
                           '\nObservation dates: {}'.format(len(obs_dates_with_distinctive_allocation),
                                                            obs_dates_with_distinctive_allocation))
        return weight_df

    @staticmethod
    def ask_user_for_weight(message: str) -> float:
        while True:
            try:
                weight_from_user = float(input(message))
            except ValueError:
                pass
            else:
                return weight_from_user


class VolatilityWeight(_PriceBasedProportionalWeight):
    """Class definition of VolatilityWeight. Subclass of _PriceBasedProportionalWeight."""

    def __init__(self, volatility_observation_period: {int, list}, inversely: bool = True, signal_df: pd.DataFrame = None,
                 total_return: bool = True, currency: str = None, price_obs_freq: {str, int}=None):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency,
                         price_obs_freq=price_obs_freq, inversely=inversely)
        self.volatility_observation_period = volatility_observation_period
        if isinstance(volatility_observation_period, int):
            self._observation_buffer = volatility_observation_period + 10
        else:
            self._observation_buffer = max(volatility_observation_period) + 10

    def _get_dataframe(self):
        price = self._get_price_df()
        volatility = realized_volatility(multivariate_price_df=price, vol_lag=self.volatility_observation_period)
        return volatility

    def get_weight_desc(self):
        if self.inversely:
            return 'weight is inversely proportional to realized volatility'
        else:
            return 'weight is proportional to realized volatility'


# TODO add minimum variance and mean variance optimized weights
class MinimumVarianceWeight(_PriceBasedWeight):
    pass



