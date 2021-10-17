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
from database.config_database import __MY_DATABASE_NAME__
from dataframe_tools import merge_two_dataframes_as_of
from financial_analysis.finance_tools import realized_volatility
from financial_analysis.financial_optimizers import theoretical_minimum_variance_portfolio_weights, \
    minimum_variance_portfolio_weights_with_constraints


# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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

    def _get_eligible_tickers_list(self):
        signal_df = self.signal_df.copy()
        signal_df[~signal_df.isnull()] = 1
        eligible_tickers_list = [list(signal_df.iloc[i, :][signal_df.iloc[i, :] == 1].index)
                                 for i in range(signal_df.shape[0])]
        return eligible_tickers_list

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


class EqualWeight(Weight):
    """Class definition of EqualWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame = None, net_zero_exposure: bool = False,
                 max_instrument_weight: float = None,
                 min_instrument_weight: float = None):
        """
        If net_zero_exposure is True the equal weighting is separately done for the long and short positions making the
        total net exposure zero in case short positions are allowed with no constraints. If False, the weights are
        calculated as equal weights regardless if it is a long or short
        position.
        :param signal_df: DataFrame
        :param net_zero_exposure: bool
        :param max_instrument_weight: float
        :param min_instrument_weight: float
        """
        super().__init__(signal_df=signal_df, max_instrument_weight=max_instrument_weight,
                         min_instrument_weight=min_instrument_weight)
        self.net_zero_exposure = net_zero_exposure

    def _calculate_weight(self):
        # calculate the number of non-NaN columns per row
        if self.net_zero_exposure:
            if com.count_not_none(self.max_instrument_weight, self.min_instrument_weight) > 0:
                logger.warning('net zero exposure is not guaranteed since constraints are specified (max/min weights)')
            positive_columns = self.signal_df[self.signal_df > 0.0].count(axis=1)
            negative_columns = self.signal_df[self.signal_df < 0.0].count(axis=1)
            weight_df = self.signal_df.copy().replace(np.nan, 0)
            weight_df[weight_df > 0.0] = weight_df[weight_df > 0.0].divide(positive_columns, axis=0)
            weight_df[weight_df < 0.0] = weight_df[weight_df < 0.0].divide(negative_columns, axis=0)
        else:
            numeric_columns = self.signal_df.count(axis=1)
            weight_df = self.signal_df.divide(numeric_columns, axis=0).replace(np.nan, 0)
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
                msg = 'Enter weight(%) for {} (#{} of {} tickers) when signal = {}: '.format(ticker, counter + 1,
                                                                                             self.signal_df.shape[0],
                                                                                             signal_value)
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


class _ProportionalWeight(Weight):
    """Class definition of _ProportionalWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame, inversely: bool, max_instrument_weight: {float, None},
                 min_instrument_weight: {float, None}):
        super().__init__(signal_df=signal_df, max_instrument_weight=max_instrument_weight,
                         min_instrument_weight=min_instrument_weight)
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

    def __init__(self, signal_df: pd.DataFrame, max_instrument_weight: {float, None},
                 min_instrument_weight: {float, None}):
        Weight.__init__(self, signal_df=signal_df, max_instrument_weight=max_instrument_weight,
                        min_instrument_weight=min_instrument_weight)
        self._financial_database_handler = FinancialDatabase(__MY_DATABASE_NAME__)

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def financial_database_handler(self):
        # make financial_database_handler read-only
        return self._financial_database_handler


class _PriceBasedWeight(_FinancialDatabaseDependentWeight):
    """Class definition of _PriceBasedSignal. Subclass of _FinancialDatabaseDependentWeight."""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int}, non_daily_price_obs: bool,
                 observation_window: {int, list, None}, clean_data: bool, max_instrument_weight: {float, None}, min_instrument_weight: {float, None}):
        _FinancialDatabaseDependentWeight.__init__(self, signal_df=signal_df,
                                                   max_instrument_weight=max_instrument_weight,
                                                   min_instrument_weight=min_instrument_weight)
        self.total_return = total_return
        self.currency = currency
        self._weekday_i_dict = {'mon': 0, 'monday': 0, 'tuesday': 1, 'tue': 1, 'wed': 2, 'wednesday': 2, 'thursday': 3,
                                'thu': 4, 'friday': 5, 'fri': 5}
        self.price_obs_freq = price_obs_freq
        self.non_daily_price_obs = non_daily_price_obs
        self.observation_window = observation_window
        self.clean_data = clean_data

    def _get_start_end_date(self):
        """
        Returns a tuple with the start and end dates used when downloading the underlying price. A buffer is added in
        order to be able to observe enough prices to make the necessary calculations
        :return: tuple
        """
        if self.signal_df is None:
            raise ValueError('signal_df needs to be specified.')
        if self.observation_window is None:  # assumes you are starting at the oldest possible date
            return None, max(self.signal_df.index)
        elif isinstance(self.observation_window, list):  # find the largest observation window
            bday_shift = max(self.observation_window)
        else:
            bday_shift = self.observation_window

        # increase the business day shift with the observation frequency
        if isinstance(self.price_obs_freq, str):
            bday_shift += 5  # assumes a weekly frequency
        else:
            bday_shift += self.price_obs_freq - 1

        # add a buffer equal to x1.25 of the original business day shift with a buffer
        bday_shift += max(bday_shift * 0.25, 15)

        # in case the price is only observed over a specific business day interval, adjust the shift accordingly
        if isinstance(self.price_obs_freq, str):
            bday_shift *= 5  # assumes weekly observations
        elif self.non_daily_price_obs:
            bday_shift *= self.price_obs_freq
        return min(self.signal_df.index) - BDay(int(bday_shift)), max(self.signal_df.index)

    def _get_raw_price_df(self) -> pd.DataFrame:
        """
        Load the historical prices from a financial database. In case prices are observed in a non daily interval, roll
        the latest available prices forward if applicable
        :return: pd.DataFrame
        """
        # retrieve the appropriate start and end dates
        start_date, end_date = self._get_start_end_date()

        # download the underlying price data
        if self.total_return:
            price = self.financial_database_handler.get_total_return_df(tickers=list(self.signal_df),
                                                                        start_date=start_date, end_date=end_date,
                                                                        withholding_tax=0, currency=self.currency)
        else:
            price = self.financial_database_handler.get_close_price_df(tickers=list(self.signal_df), start_date=start_date,
                                                                       end_date=end_date, currency=self.currency)

        # roll prices forward if applicable
        if (self.price_obs_freq != 1 and self.non_daily_price_obs or isinstance(self.price_obs_freq, str)) and self.clean_data:
            price = self._clean_price_df(raw_price_df=price, ffill=True, bfill=False)
            price = self._filter_price_based_on_frequency(price_df=price)
        return price

    def _get_price_df(self):
        """
        Load the raw price and clean it up if applicable. Then filter prices based on the specified observation
        frequency
        :return: pd.DataFrame
        """
        price = self._get_raw_price_df()
        if self.clean_data:
            price = self._clean_price_df(raw_price_df=price, ffill=True, bfill=False)
        if self.non_daily_price_obs or isinstance(self.price_obs_freq, str):
            price = self._filter_price_based_on_frequency(price_df=price)
        return price

    def _get_price_return_df(self) -> pd.DataFrame:
        """
        Calculate the price returns based on the specified observation frequency
        :return: pd.DataFrame
        """
        # get the price and calculate the return
        raw_price_df = self._get_raw_price_df()
        clean_price_df = self._clean_price_df(raw_price_df=raw_price_df, ffill=True, bfill=False)
        if self.non_daily_price_obs or isinstance(self.price_obs_freq, str):
            return_df = clean_price_df.pct_change()
        else:
            return_df = clean_price_df.pct_change(self.price_obs_freq)

        # clean price return data if applicable
        if self.clean_data:
            # TODO too aggressive?
            return_df.replace(np.nan, 0, inplace=True)
        else:
            # set to nan where the raw price equals nan
            nan_or_1 = raw_price_df.copy()
            nan_or_1[~nan_or_1.isnull()] = 1  # set all non NaN to 1
            return_df *= nan_or_1.values
        return return_df

    def _get_price_df_list(self):
        price_df = self._get_price_df()
        return self._get_data_list(multivariate_df=price_df)

    def _get_price_return_df_list(self):
        price_return_df = self._get_price_return_df()
        return self._get_data_list(multivariate_df=price_return_df)

    def _get_data_list(self, multivariate_df: pd.DataFrame):
        """
        Returns a list of DataFrames. The ith DataFrame represents the data to be used for the signal observed on the
        ith observation date of the ith set of eligible tickers
        :param multivariate_df: pd.DataFrame
        :return: list of pd.DataFrame
        """

        # each element (which is a list of str) in this list represents the tickers that has a defined signal for the
        # ith observation date
        eligible_tickers_list = self._get_eligible_tickers_list()

        # adjust the observation date to only include dates that exists in the price DataFrame
        signal_obs_calendar = self.signal_df.index
        data_obs_calendar = multivariate_df.index
        adj_obs_date_i_list = [data_obs_calendar.get_loc(obs_date, method='ffill')
                               for obs_date in signal_obs_calendar]

        # this looks messy but all we are doing is to create a list of DataFrames where the column and row selection is
        # dictated by the adjusted observation dates, eligible tickers and observation window
        num_obs_dates = len(signal_obs_calendar)
        if self.observation_window is None:
            # use no lower limit for the observation (starting from the oldest available observation date)
            df_list = [multivariate_df.iloc[:adj_obs_date_i_list[i] + 1, :][eligible_tickers_list[i]]
                       for i in range(num_obs_dates)]
        elif isinstance(self.observation_window, int):
            df_list = [multivariate_df.iloc[adj_obs_date_i_list[i] - self.observation_window + 1:
                                            adj_obs_date_i_list[i] + 1, :][eligible_tickers_list[i]]
                       for i in range(num_obs_dates)]
        else:
            raise ValueError('observation_window needs to be an int or None')
        return df_list

    def _filter_price_based_on_frequency(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select prices based on a particular weekday (represented by a str) or a specific interval (int)
        :param price_df: pd.DataFrame
        :return: pd.DataFrame
        """
        # filter out rows if you have specified certain observation intervals or weekdays
        if isinstance(self.price_obs_freq, str):
            price_df = price_df[price_df.index.weekday == self._weekday_i_dict[self.price_obs_freq]]
        elif isinstance(self.price_obs_freq, int):
            # sort index in descending order. this is done to have the count start from the latest observation date
            price_df = price_df.sort_index(ascending=False).iloc[::self.price_obs_freq, :].sort_index()
        return price_df

    def get_annualization_factor(self) -> float:
        """
        Calculate an annualization factor as the ratio of 252 and the return lag.
        :return: float
        """

        if isinstance(self.price_obs_freq, str):
            annualization_factor = 252 / 5
        else:
            annualization_factor = 252 / self.price_obs_freq
        return annualization_factor

    @staticmethod
    def _clean_price_df(raw_price_df: pd.DataFrame, ffill: bool, bfill: bool):
        """
        Replaces nan with other available prices
        :param raw_price_df: pd.DataFrame
        :param ffill: bool -> if price is nan, replace it with the closest price available in the past
        :param bfill: bool -> if price is nan, replace it with the closest price available in the future
        :return:
        """
        clean_price_df = raw_price_df.copy()
        if ffill:
            clean_price_df.fillna(method='ffill', inplace=True)

        if bfill:
            clean_price_df.fillna(method='bfill', inplace=True)
        return clean_price_df

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
            self._price_obs_freq = 1
        else:
            raise ValueError("price_obs_freq needs to be an int larger or equal to 1 or a string equal to '%s'."
                             % "' or '".join(self._weekday_i_dict.keys()))


class _PriceBasedProportionalWeight(_PriceBasedWeight, _ProportionalWeight):
    """Class definition of _PriceBasedProportionalWeight. Subclass of _PriceBasedWeight and
    _ProportionalWeight."""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int},
                 non_daily_price_obs: bool, observation_window: {int, list, None}, clean_data: bool,
                 inversely: bool, max_instrument_weight: {float, None}, min_instrument_weight: {float, None}):
        _PriceBasedWeight.__init__(self, signal_df=signal_df, total_return=total_return, currency=currency,
                                   price_obs_freq=price_obs_freq, non_daily_price_obs=non_daily_price_obs,
                                   observation_window=observation_window, clean_data=clean_data,
                                   max_instrument_weight=max_instrument_weight, min_instrument_weight=min_instrument_weight)
        _ProportionalWeight.__init__(self, signal_df=signal_df, inversely=inversely,
                                     max_instrument_weight=max_instrument_weight,
                                     min_instrument_weight=min_instrument_weight)


class VolatilityWeight(_PriceBasedProportionalWeight):
    """Class definition of VolatilityWeight. Subclass of _PriceBasedProportionalWeight."""

    def __init__(self, volatility_observation_period: {int, list}, inversely: bool = True,
                 signal_df: pd.DataFrame = None,
                 total_return: bool = True, currency: str = None, price_obs_freq: {str, int}=None,
                 non_daily_price_obs: bool = False, max_instrument_weight: float = None, min_instrument_weight: float = None,
                 clean_data: bool = False):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency,
                         price_obs_freq=price_obs_freq, non_daily_price_obs=non_daily_price_obs,
                         observation_window=volatility_observation_period, clean_data=clean_data,
                         inversely=inversely, max_instrument_weight=max_instrument_weight,
                         min_instrument_weight=min_instrument_weight)
        self.volatility_observation_period = volatility_observation_period

    def _get_dataframe(self):
        annualization_factor = self.get_annualization_factor()
        price = self._get_price_df()
        if self.non_daily_price_obs or isinstance(self.price_obs_freq, str):
            return_lag = 1
        else:
            return_lag = self.price_obs_freq
        volatility = realized_volatility(multivariate_price_df=price, vol_lag=self.volatility_observation_period,
                                         annualized_factor=annualization_factor, return_lag=return_lag)
        return volatility

    def get_weight_desc(self):
        if self.inversely:
            return 'weight is inversely proportional to realized volatility'
        else:
            return 'weight is proportional to realized volatility'

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def volatility_observation_period(self):
        return self.observation_window

    @volatility_observation_period.setter
    def volatility_observation_period(self, volatility_observation_period: {int, list}):
        self.observation_window = volatility_observation_period

class _OptimizedWeight(_PriceBasedWeight):
    """Class definition of _OptimizedWeight"""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, max_instrument_weight: {float, None},
                 min_instrument_weight: {float, None}, max_total_weight: {float, None}, min_total_weight: {float, None},
                 price_return_lag: int, non_daily_price_obs: bool, observation_window: int, calculate_mean_returns: bool,
                 has_analytical_solution: bool, clean_data: bool):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency, price_obs_freq=price_return_lag,
                         non_daily_price_obs=non_daily_price_obs, clean_data=clean_data, observation_window=observation_window,
                         max_instrument_weight=max_instrument_weight, min_instrument_weight=min_instrument_weight)
        self.max_total_weight = max_total_weight
        self.min_total_weight = min_total_weight
        self.calculate_mean_returns = calculate_mean_returns
        self.has_analytical_solution = has_analytical_solution

    def _optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}, initial_guess: np.array) -> np.array:
        raise ValueError('an optimizer has not been specified')

    def _theoretical_optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}) -> np.array:
        raise ValueError('an optimizer with an analytical solution has not been specified')

    def _calculate_weight(self):
        """
        Return a DataFrame with the same index and columns as the signal DataFrame containing the optimized weights.
        A list of price return DataFrames is used to calculate a list of covariance matrices and mean return vectors
        when applicable. If specified and when applicable, the script will use an analytical solution of the optimization
        problem.
        :return:
        """
        # retrieve a list of DataFrames containing price returns
        return_df_list = self._get_price_return_df_list()

        # using the price returns calculate the inputs to be used in the optimizer
        # inputs should all be numpy arrays
        cov_matrix_list = self._get_covariance_matrix_list(return_df_list)
        if self.calculate_mean_returns:
            mean_return_list = self._get_mean_return_array_list(return_df_list)
        else:
            mean_return_list = self.signal_df.shape[0] * [None]

        # only use the theoretical optimizer when there are no constraints and if an analytical solution is available
        use_theoretical_optimizer = (com.count_not_none(self.min_instrument_weight, self.max_instrument_weight) == 0) \
                                    and self.min_total_weight == self.max_total_weight and self.has_analytical_solution

        # list of list of eligible tickers (i.e. having a defined signal) for each signal observation date
        eligible_tickers_list = self._get_eligible_tickers_list()

        # loop through each observation date and use the optimizer to calculate the weights
        prev_eligible_tickers = eligible_tickers_list[0]
        optimized_weight_list = []  # the solved weight arrays are stored here
        self._counter = 0  # used to display progress
        self._total = len(return_df_list)  # used to display progress
        for i in range(self.signal_df.shape[0]):
            if use_theoretical_optimizer:
                # no initial guess is needed when using a theoretical/analytical solution
                # however since _theoretical_optimizer passes on the
                optimized_weight = self._theoretical_optimizer(
                    cov_matrix=cov_matrix_list[i],
                    mean_returns=mean_return_list[i]
                    )
            else:
                # an initial guess is needed when using an optimizer with constraints
                # this initial guess is the previous solved weights except at the start or when the instruments changes
                new_eligible_tickers = eligible_tickers_list[i]
                if i == 0 or prev_eligible_tickers != new_eligible_tickers:
                    initial_guess = None  # equal weights by default inside the optimization function
                else:
                    initial_guess = optimized_weight_list[i - 1]  # previous solved weights
                optimized_weight = self._optimizer(
                    cov_matrix=cov_matrix_list[i],
                    mean_returns=mean_return_list[i],
                    initial_guess=initial_guess
                )
                prev_eligible_tickers = new_eligible_tickers

                # display progress
                self._counter += 1
                logger.info('progress: {}%...'.format(round(100 * self._counter / self._total, 2)))
            optimized_weight_list.append(optimized_weight)

        # reformat result as a DataFrame
        optimized_weight_df = pd.DataFrame(np.zeros((self.signal_df.shape[0], self.signal_df.shape[1])),
                                           index=self.signal_df.index, columns=self.signal_df.columns)
        obs_calendar = self.signal_df.index
        for i in range(self.signal_df.shape[0]):
            obs_date = obs_calendar[i]
            eligible_tickers = eligible_tickers_list[i]
            optimized_weight_df.loc[obs_date, eligible_tickers] = optimized_weight_list[i]
        return optimized_weight_df

    @staticmethod
    def _get_covariance_matrix_list(return_df_list: list):
        """
        Calculates a covariance matrix for each price return DataFrame in the given list. Returns a list of numpy arrays
        :param return_df_list: list of DataFrames
        :return: list of numpy arrays
        """
        cov_matrix_list = [np.cov(np.transpose(return_df.values).astype(float)) for return_df in return_df_list]
        return cov_matrix_list

    @staticmethod
    def _get_mean_return_array_list(return_df_list: list):
        """
        Calculates the mean returns for each price return DataFrame in the given list. Returns a list of numpy arrays
        :param return_df_list: list of DataFrames
        :return: list of numpy arrays
        """
        mean_return_list = [return_df.mean().values for return_df in return_df_list]
        return mean_return_list

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def price_return_lag(self):
        return self.price_obs_freq

    @price_return_lag.setter
    def price_return_lag(self, price_return_lag: {int, str}):
        self.price_obs_freq = price_return_lag


class MinimumVarianceWeight(_OptimizedWeight):
    """Class definition for MinimumVarianceWeight"""

    def __init__(self, observation_window: int, price_return_lag: int = 1, max_instrument_weight: float = None,
                 min_instrument_weight: float = None, total_allocation: float = 1.0, signal_df: pd.DataFrame = None,
                 total_return: bool = False, currency: str = None, non_daily_price_obs: bool = False,
                 clean_data: bool = True):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency,
                         max_instrument_weight=max_instrument_weight, min_instrument_weight=min_instrument_weight,
                         max_total_weight=total_allocation, min_total_weight=total_allocation,
                         price_return_lag=price_return_lag, observation_window=observation_window,
                         calculate_mean_returns=False, has_analytical_solution=True,
                         non_daily_price_obs=non_daily_price_obs, clean_data=clean_data)
        self._total_allocation = total_allocation

    def _optimizer(self, cov_matrix: np.array, mean_returns: np.array, initial_guess: np.array):
        annualization_factor = self.get_annualization_factor()
        return minimum_variance_portfolio_weights_with_constraints(
            covariance_matrix=cov_matrix, initial_guess=initial_guess, max_total_weight=self.total_allocation,
            max_instrument_weight=self.max_instrument_weight, min_instrument_weight=self.min_instrument_weight,
            annualizing_factor=annualization_factor
        )

    def _theoretical_optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}):
        return theoretical_minimum_variance_portfolio_weights(
            covariance_matrix=cov_matrix, allocation=self.total_allocation
        )

    @property
    def total_allocation(self):
        return self._total_allocation

    @total_allocation.setter
    def total_allocation(self, total_allocation: float):
        self._total_allocation = total_allocation
        self.max_total_weight = total_allocation
        self.min_total_weight = total_allocation


