"""
strategy_overlay.py
"""

import pandas as pd
import numpy as np

# my modules
from dataframe_tools import merge_two_dataframes_as_of
from financial_analysis.finance_tools import convert_daily_return_to_daily_price_df, realized_volatility, beta
from database.financial_database import FinancialDatabase
from database.config_database import __MY_DATABASE_NAME__


class _Overlay:
    """Class definition of _Overlay"""

    def __init__(self, max_value: float, min_value: float, weight_obs_lag: int = 1, scaling_factor_lag: int = 0,
                 avg_smoothing_lag: int = 1):
        self.max_value = max_value
        self.min_value = min_value
        self.weight_obs_lag = weight_obs_lag
        self.scaling_factor_lag = scaling_factor_lag
        self.avg_smoothing_lag = avg_smoothing_lag

    def get_return_weight_tuple_after_scaling(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd):
        self._check_inputs(multivariate_daily_return_df, multivariate_daily_weight_df)
        adj_result = self._adjustment_method(multivariate_daily_return_df, multivariate_daily_weight_df)
        if list(adj_result[0]) == list(adj_result[1]):
            return adj_result
        else:
            raise ValueError('the column headers of the return and weight DataFrames are not the same')

    def _apply_constraints(self, scaling_factor_df: pd.DataFrame):
        # apply the scaling to each column
        result = scaling_factor_df.copy()
        for col_name in list(result):
            # apply the lambda function that adds the constraints
            result.loc[:, col_name] = result[col_name].apply(
                lambda x: max(self.min_value, min(self.max_value, x))
                if not np.isnan(x)
                else np.nan
            )
        return result

    def _apply_smoothing(self, scaling_factor_df: pd.DataFrame):
        if self.avg_smoothing_lag == 1:
            return scaling_factor_df
        else:
            scaling_factor_df.rolling(window=self.avg_smoothing_lag).mean()

    def _adjustment_method(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd.DataFrame)\
            ->(pd.DataFrame, pd.DataFrame):
        raise ValueError('_adjustment_method is not defined')

    def _check_inputs(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd.DataFrame):
        return

    def _calculate_strategy_return(self, multivariate_daily_return_df: pd.DataFrame,
                                   multivariate_daily_weight_df: pd.DataFrame, sum_columns: bool = True) -> pd.DataFrame:
        strategy_df = multivariate_daily_return_df * multivariate_daily_weight_df.shift(self.weight_obs_lag)
        if sum_columns:
            # find the index where all columns are nan
            all_col_nan = strategy_df.isnull().sum(axis=1) == strategy_df.shape[1]
            strategy_df = pd.DataFrame({'strategy_return': strategy_df.sum(axis=1).values},
                                       index=multivariate_daily_return_df.index)
            strategy_df[all_col_nan.values] = np.nan
        return strategy_df

    @staticmethod
    def get_desc():
        return 'none'

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self.get_desc())

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def weight_obs_lag(self):
        return self._weight_obs_lag

    @weight_obs_lag.setter
    def weight_obs_lag(self, weight_obs_lag: int):
        if weight_obs_lag >= 1:
            self._weight_obs_lag = weight_obs_lag
        else:
            raise ValueError('weight_obs_lag needs to be greater or equal to 1 in order to make the strategy replicable')

    @property
    def scaling_factor_lag(self):
        return self._scaling_factor_lag

    @scaling_factor_lag.setter
    def scaling_factor_lag(self, scaling_factor_lag: int):
        if scaling_factor_lag >= 0:
            self._scaling_factor_lag = scaling_factor_lag
        else:
            raise ValueError(
                'scaling_factor_lag needs to be greater or equal to 0 in order to make the strategy replicable')

    @property
    def avg_smoothing_lag(self):
        return self._avg_smoothing_lag

    @avg_smoothing_lag.setter
    def avg_smoothing_lag(self, avg_smoothing_lag: int):
        if avg_smoothing_lag >= 1:
            self._avg_smoothing_lag = avg_smoothing_lag
        else:
            raise ValueError('avg_smoothing_lag needs to be greater or equal to 1')


class VolatilityControl(_Overlay):
    """Class definition of VolatilityControl"""

    def __init__(self, vol_control_level: float, vol_lag: {int, list, tuple}, max_risky_weight: float = 1.0, min_risky_weight: float = 0.0, weight_obs_lag: int = 1,
                 scaling_factor_lag: int = 0, avg_smoothing_lag: int = 1):
        super().__init__(max_value=max_risky_weight, min_value=min_risky_weight, weight_obs_lag=weight_obs_lag,
                         scaling_factor_lag=scaling_factor_lag, avg_smoothing_lag=avg_smoothing_lag)
        self.vol_control_level = vol_control_level
        self.vol_lag = vol_lag

    def _adjustment_method(self, multivariate_daily_return_df, multivariate_daily_weight_df)-> (pd.DataFrame, pd.DataFrame):
        """
        Calculate a scaling factor based on the volatility target level divided by the realised volatility of the
        strategy return subject to the constraints
        :param multivariate_daily_return_df: pd.DataFrame
        :param multivariate_daily_weight_df: pd.DataFrame
        :return: (pd.DataFrame, pd.DataFrame)
        """
        strategy_return_df = self._calculate_strategy_return(multivariate_daily_return_df, multivariate_daily_weight_df)
        realized_vol = realized_volatility(multivariate_return_df=strategy_return_df, vol_lag=self.vol_lag)

        # calculate the scaling factor as the VT divided by the realized volatility and apply the constraints
        scaling_factor_df = realized_vol.copy()
        for col in scaling_factor_df.columns:
            scaling_factor_df[col].values[:] = self.vol_control_level
        scaling_factor_df = scaling_factor_df.divide(realized_vol)
        scaling_factor_df = self._apply_constraints(scaling_factor_df)

        # new weights
        multivariate_daily_weight_df *= scaling_factor_df.shift(self.scaling_factor_lag).values
        multivariate_daily_weight_df = self._apply_smoothing(multivariate_daily_weight_df)
        return multivariate_daily_return_df, multivariate_daily_weight_df

    def get_desc(self):
        return 'VT={}%, max_risky_weight={}%, min_risky_weight={}%'.format(100 * self.vol_control_level,
                                                                           100 * self.max_value, 100 * self.min_value)

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def vol_control_level(self):
        return self._vol_control_level

    @vol_control_level.setter
    def vol_control_level(self, vol_control_level: float):
        if vol_control_level > 0:
            self._vol_control_level = vol_control_level
        else:
            raise ValueError('vol_control_level needs to be strictly larger than 0')

    @property
    def max_risky_weight(self):
        return self.max_value

    @max_risky_weight.setter
    def max_risky_weight(self, max_risky_weight: float):
        self.max_value = max_risky_weight

    @property
    def min_risky_weight(self):
        return self.min_value

    @min_risky_weight.setter
    def min_risky_weight(self, min_risky_weight: float):
        self.min_value = min_risky_weight


class _DatabaseDependentOverlay(_Overlay):
    """Class definition of _DatabaseDependentOverlay"""

    def __init__(self, max_value: float, min_value: float, weight_obs_lag: int = 1, scaling_factor_lag: int = 0,
                 avg_smoothing_lag: int = 1):
        super().__init__(max_value=max_value, min_value=min_value, weight_obs_lag=weight_obs_lag,
                         scaling_factor_lag=scaling_factor_lag, avg_smoothing_lag=avg_smoothing_lag)
        self._financial_db_handler = FinancialDatabase(__MY_DATABASE_NAME__)

    @property
    def financial_db_handler(self):
        return self._financial_db_handler


class BetaHedge(_DatabaseDependentOverlay):
    """Class definition of _DatabaseDependentOverlay"""

    def __init__(self, beta_instrument_ticker: str, beta_lag: {int, list, tuple},
                 beta_instrument_total_return: bool = False, return_lag: int = 1,
                 max_beta_value: float = 999.0, min_beta_value: float = 0.0, scaling_factor_lag: int = 0,
                 avg_smoothing_lag: int = 1):
        super().__init__(max_value=max_beta_value, min_value=min_beta_value, scaling_factor_lag=scaling_factor_lag,
                         avg_smoothing_lag=avg_smoothing_lag)
        self.beta_instrument_ticker = beta_instrument_ticker
        self.beta_lag = beta_lag
        self.beta_instrument_total_return = beta_instrument_total_return
        self.return_lag = return_lag

    def _check_beta_price_availability(self, start_date, end_date, available_calendar):
        error_msg = 'Not enough data for the beta instrument. Please refresh {}.\nNeeds data from {} and {} '\
            .format(self.beta_instrument_ticker, str(start_date)[:10], str(end_date)[:10])
        if start_date < available_calendar[0]:
            error_msg += 'but data is only available from {}.'.format(str(available_calendar[0])[:10])
            raise ValueError(error_msg)
        elif end_date > available_calendar[-1]:
            error_msg += 'but data is only available up to {}.'.format(str(available_calendar[-1])[:10])
            raise ValueError(error_msg)
        else:
            return

    def _adjustment_method(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd.DataFrame)->(pd.DataFrame, pd.DataFrame):
        """
        Calculate the beta as a weighted average of the betas for each instrument. The daily return and weight of the
        beta instrument gets added to the return and weight DataFrames.
        :param multivariate_daily_return_df: pd.DataFrame
        :param multivariate_daily_weight_df: pd.DataFrame
        :return: (pd.DataFrame, pd.DataFrame)
        """
        # download and check availability of the price of the relevant beta instrument
        beta_price_df = self._get_beta_price(multivariate_daily_return_df=multivariate_daily_return_df)

        # calculate the 'strategy beta' as the weighted average of all the instrument betas
        strategy_beta_df = self._get_strategy_beta(multivariate_daily_return_df=multivariate_daily_return_df,
                                                   multivariate_daily_weight_df=multivariate_daily_weight_df,
                                                   beta_price_df=beta_price_df)

        # apply the constraints
        strategy_beta_df = self._apply_constraints(scaling_factor_df=strategy_beta_df)
        strategy_beta_df *= -1  # make it a short position
        strategy_beta_df = strategy_beta_df.shift(self.scaling_factor_lag)
        strategy_beta_df = self._apply_smoothing(strategy_beta_df)  # smooth the results if applicable

        # when beta is NaN make all other weights NaN
        beta_nan_else_1 = strategy_beta_df.copy()
        beta_nan_else_1[~beta_nan_else_1.isnull()] = 1
        multivariate_daily_weight_df *= beta_nan_else_1.values

        # add the beta instrument and the beta weight to the daily return and weight DataFrames
        multivariate_daily_return_df = merge_two_dataframes_as_of(multivariate_daily_return_df, beta_price_df.pct_change())
        multivariate_daily_weight_df = merge_two_dataframes_as_of(multivariate_daily_weight_df, strategy_beta_df)
        return multivariate_daily_return_df, multivariate_daily_weight_df

    def _get_beta_price(self, multivariate_daily_return_df: pd.DataFrame):
        # download and check availability of the price of the relevant beta instrument
        start_date = multivariate_daily_return_df.index[0]
        end_date = multivariate_daily_return_df.index[-1]
        if self.beta_instrument_total_return:
            beta_price_df = self.financial_db_handler.get_total_return_df(tickers=self.beta_instrument_ticker,
                                                                          start_date=start_date, end_date=end_date)
        else:
            beta_price_df = self.financial_db_handler.get_close_price_df(tickers=self.beta_instrument_ticker,
                                                                         start_date=start_date, end_date=end_date)
        self._check_beta_price_availability(start_date, end_date, beta_price_df.index)

        # name of the column (the name in the weight and return have to be the same)
        beta_col_name = 'beta_instrument ({})'.format(list(beta_price_df)[0])
        beta_price_df.columns = [beta_col_name]
        return beta_price_df

    def _get_beta_per_instrument(self, multivariate_daily_return_df: pd.DataFrame, beta_price_df: pd.DataFrame):
        # if the return lag is anything but 1 use daily performance data for the beta calculation
        if self.return_lag > 1:
            # convert the daily prices to daily observed performance date
            daily_perf_df = convert_daily_return_to_daily_price_df(
                multivariate_daily_return_df=multivariate_daily_return_df)

            # calculate the beta for each instrument
            instrument_beta_df = beta(multivariate_price_df=daily_perf_df, beta_price_df=beta_price_df,
                                      beta_lag=self.beta_lag, return_lag=self.return_lag)
        else:
            # calculate the beta for each instrument
            instrument_beta_df = beta(multivariate_return_df=multivariate_daily_return_df, beta_price_df=beta_price_df,
                                      beta_lag=self.beta_lag)
        return instrument_beta_df

    def _get_strategy_beta(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd.DataFrame,
                           beta_price_df: pd.DataFrame):
        # calculate the beta for each individual instrument and return a merged DataFrame
        instrument_beta_df = self._get_beta_per_instrument(multivariate_daily_return_df=multivariate_daily_return_df,
                                                           beta_price_df=beta_price_df)

        # calculate the 'strategy beta' as the weighted average of all the instrument betas
        weighted_instrument_beta_df = instrument_beta_df * multivariate_daily_weight_df.values
        beta_col_name = beta_price_df.columns.values[0]
        strategy_beta_df = pd.DataFrame({beta_col_name: weighted_instrument_beta_df.sum(axis=1, skipna=False).values},
                                        index=weighted_instrument_beta_df.index)
        return strategy_beta_df

    def get_desc(self):
        return 'beta_instrument={}, max_beta={}%, min_beta={}%'.format(self.beta_instrument_ticker,
                                                                       100 * self.max_value, 100 * self.min_value)

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def beta_instrument_ticker(self):
        return self._beta_instrument_ticker

    @beta_instrument_ticker.setter
    def beta_instrument_ticker(self, beta_instrument_ticker: str):
        self._beta_instrument_ticker = beta_instrument_ticker.upper()

    @property
    def max_beta_value(self):
        return self.max_value

    @max_beta_value.setter
    def max_beta_value(self, max_beta_value: float):
        self.max_value = max_beta_value

    @property
    def min_beta_value(self):
        return self.min_value

    @min_beta_value.setter
    def min_beta_value(self, min_beta_value: float):
        self.min_value = min_beta_value


