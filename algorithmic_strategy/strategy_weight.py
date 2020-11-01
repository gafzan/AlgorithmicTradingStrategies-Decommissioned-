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
from financial_analysis.financial_optimizers import theoretical_minimum_variance_portfolio_weights, \
    minimum_variance_portfolio_weights_with_constraints

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
        self._financial_database_handler = FinancialDatabase(my_database_name)

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def financial_database_handler(self):
        # make financial_database_handler read-only
        return self._financial_database_handler


class _PriceBasedWeight(_FinancialDatabaseDependentWeight):
    """Class definition of _PriceBasedSignal. Subclass of _FinancialDatabaseDependentWeight."""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int},
                 max_instrument_weight: {float, None}, min_instrument_weight: {float, None}):
        _FinancialDatabaseDependentWeight.__init__(self, signal_df=signal_df, max_instrument_weight=max_instrument_weight,
                                                   min_instrument_weight=min_instrument_weight)
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
            # TODO maybe don't start from the latest available date?
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
                 inversely: bool, max_instrument_weight: {float, None}, min_instrument_weight: {float, None}):
        _PriceBasedWeight.__init__(self, signal_df=signal_df, total_return=total_return, currency=currency,
                                   price_obs_freq=price_obs_freq, max_instrument_weight=max_instrument_weight,
                                   min_instrument_weight=min_instrument_weight)
        _ProportionalWeight.__init__(self, signal_df=signal_df, inversely=inversely,
                                     max_instrument_weight=max_instrument_weight,
                                     min_instrument_weight=min_instrument_weight)


class EqualWeight(Weight):
    """Class definition of EqualWeight. Subclass of Weight."""

    def __init__(self, signal_df: pd.DataFrame = None, net_zero_exposure: bool = False, max_instrument_weight: float = None,
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
        super().__init__(signal_df=signal_df, max_instrument_weight=max_instrument_weight, min_instrument_weight=min_instrument_weight)
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
                 total_return: bool = True, currency: str = None, price_obs_freq: {str, int}=None,
                 max_instrument_weight: float = None, min_instrument_weight: float = None):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency,
                         price_obs_freq=price_obs_freq, inversely=inversely, max_instrument_weight=max_instrument_weight,
                         min_instrument_weight=min_instrument_weight)
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


class _OptimizedWeight(_PriceBasedWeight):
    """Class definition of _OptimizedWeight"""

    def __init__(self, signal_df: pd.DataFrame, total_return: bool, currency: str, max_instrument_weight: {float, None},
                 min_instrument_weight: {float, None}, max_total_weight: {float, None}, min_total_weight: {float, None},
                 price_return_lag: int, observation_window: int, calculate_mean_returns: bool, has_analytical_solution: bool):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency, price_obs_freq=None,
                         max_instrument_weight=None, min_instrument_weight=None)
        self.op_max_i_w = max_instrument_weight
        self.op_min_i_w = min_instrument_weight
        self.max_total_w = max_total_weight
        self.min_total_w = min_total_weight
        self.price_return_lag = price_return_lag
        self.observation_window = observation_window
        self.calculate_mean_returns = calculate_mean_returns
        self._observation_buffer = observation_window + self.price_return_lag + 10
        self.has_analytical_solution = has_analytical_solution
        self.annualizing_factor = 252 / self.price_return_lag  # to be used in the non analytical optimization

    def _optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}, initial_guess: np.array) -> np.array:
        raise ValueError('an optimizer has not been specified')

    def _theoretical_optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}) -> np.array:
        raise ValueError('an optimizer with an analytical solution has not been specified')

    def _calculate_weight(self):
        signal_df = self.signal_df.copy()
        signal_df[~signal_df.isnull()] = 1  # set all non NaN to 1 (MWO are agnostic to initial strategy signals)

        obs_calendar = signal_df.index
        num_obs_dates = len(obs_calendar)

        # selection based on price history and if the ticker is eligible (based on having a valid signal)
        eligible_tickers_list = self.get_eligible_tickers_list()

        # retrieve a list of DataFrames containing price returns
        return_df_list = self._get_price_return_df_list(eligible_tickers_list=eligible_tickers_list)

        # using the price returns calculate the inputs to be used in the optimizer (correlation etc.)
        # inputs should all be numpy arrays
        cov_matrix_list = self._get_covariance_matrix_list(return_df_list)
        if self.calculate_mean_returns:
            mean_return_list = self._get_mean_return_array_list(return_df_list)
        else:
            mean_return_list = num_obs_dates * [None]

        # only use the theoretical optimizer when there are no constraints and if an analytical solution is available
        use_theoretical_optimizer = (com.count_not_none(self.op_min_i_w, self.op_max_i_w) == 0) \
                                    and self.min_total_w == self.max_total_w and self.has_analytical_solution

        # loop through each observation date and use the optimizer to calculate the weights
        prev_eligible_tickers = eligible_tickers_list[0]
        optimized_weight_list = []  # the solved weight arrays are stored here
        for i in range(num_obs_dates):
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
            optimized_weight_list.append(optimized_weight)

        # reformat result as a DataFrame
        optimized_weight_df = pd.DataFrame(np.zeros((signal_df.shape[0], signal_df.shape[1])),
                                           index=signal_df.index, columns=signal_df.columns)
        for i in range(num_obs_dates):
            obs_date = obs_calendar[i]
            eligible_tickers = eligible_tickers_list[i]
            optimized_weight_df.loc[obs_date, eligible_tickers] = optimized_weight_list[i]
        return optimized_weight_df

    # TODO why can't I use _get_eligible_tickers_list as a name of the method?
    def get_eligible_tickers_list(self):
        signal_df = self.signal_df.copy()
        signal_df[~signal_df.isnull()] = 1  # set all non NaN to 1 (MWO are agnostic to initial strategy signals)
        eligible_tickers_list = [list(signal_df.iloc[i, :][signal_df.iloc[i, :] == 1].index)
                                 for i in range(signal_df.shape[0])]
        return eligible_tickers_list

    def _get_price_return_df_list(self, eligible_tickers_list: list):
        obs_calendar = self.signal_df.index
        num_obs_dates = len(obs_calendar)
        # retrieve the price returns
        multivariate_return_df = self._get_price_return_df()

        # adjust the observation date to only include dates that exists in the price DataFrame
        daily_calendar = multivariate_return_df.index
        adj_obs_date_i_list = [daily_calendar.get_loc(obs_date, method='ffill')
                               for obs_date in obs_calendar]

        # create a list of price DataFrames to be used in the optimization
        return_df_list = [multivariate_return_df.iloc[adj_obs_date_i_list[i] - self.observation_window + 1:
                                                      adj_obs_date_i_list[i] + 1, :][eligible_tickers_list[i]]
                          for i in range(num_obs_dates)]
        return return_df_list

    def _get_price_return_df(self) -> pd.DataFrame:
        """
        retrieve the price data and calculate the price returns. returns are calculated based on prices with NaN are
        'forward filled'. returns are removed where a price was NaN
        :return: pd.DataFrame
        """
        multivariate_price_df = self._get_price_df()
        multivariate_return_df = multivariate_price_df.fillna(method='ffill').pct_change(self.price_return_lag)
        nan_or_1 = multivariate_price_df.copy()
        nan_or_1[~nan_or_1.isnull()] = 1  # set all non NaN to 1
        multivariate_return_df *= nan_or_1.values
        return multivariate_return_df

    def _get_eligible_tickers_list(self):
        pass

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


class MinimumVarianceWeight(_OptimizedWeight):
    """Class definition for MinimumVarianceWeight"""

    def __init__(self, observation_window: int, price_return_lag: int = 1, max_instrument_weight: float = None,
                 min_instrument_weight: float = None, total_allocation: float = 1.0, signal_df: pd.DataFrame = None,
                 total_return: bool = False, currency: str = None):
        super().__init__(signal_df=signal_df, total_return=total_return, currency=currency,
                         max_instrument_weight=max_instrument_weight, min_instrument_weight=min_instrument_weight,
                         max_total_weight=total_allocation, min_total_weight=total_allocation,
                         price_return_lag=price_return_lag, observation_window=observation_window,
                         calculate_mean_returns=False, has_analytical_solution=True)

    def _optimizer(self, cov_matrix: np.array, mean_returns: np.array, initial_guess: np.array):
        # error handling
        pd.DataFrame(data=cov_matrix).to_clipboard()
        return minimum_variance_portfolio_weights_with_constraints(
            covariance_matrix=cov_matrix, initial_guess=initial_guess, max_total_weight=self.max_total_w,
            max_instrument_weight=self.op_max_i_w, min_instrument_weight=self.op_min_i_w,
            annualizing_factor=self.annualizing_factor
        )

    def _theoretical_optimizer(self, cov_matrix: np.array, mean_returns: {np.array, None}):
        return theoretical_minimum_variance_portfolio_weights(
            covariance_matrix=cov_matrix, allocation=self.max_total_w
        )


def main():
    from excel_tools import load_df
    file_path = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\excel_data\signal_test.xlsx'
    signal_df = load_df(full_path=file_path)

    weight = MinimumVarianceWeight(60, signal_df=signal_df, min_instrument_weight=0)
    weight_df = weight.get_weights()
    print(weight_df)


if __name__ == '__main__':
    main()

