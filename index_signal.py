import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from financial_database import FinancialDatabase

# my own modules
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar, check_if_values_in_dataframe_are_allowed
from finance_tools import rolling_average, realized_volatility
from config_database import my_database_name


class Signal:
    """Class definition of Signal."""

    financial_database_name = r'sqlite:///C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\stock_db_v3'

    def __init__(self, ticker_list: list = None, signal_observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None):
        # either assign eligibility_df or ticker_list and signal_observation_calendar
        if not eligibility_df and ticker_list:
            self._ticker_list = ticker_list
            self._signal_observation_calendar = signal_observation_calendar
        elif eligibility_df and not (ticker_list or signal_observation_calendar):
            self._ticker_list = list(eligibility_df)
            self._signal_observation_calendar = eligibility_df.index
        else:
            raise ValueError("Need to assign ticker_list or only eligibility_df.")
        self._eligibility_df = eligibility_df
        self._financial_database_handler = FinancialDatabase(my_database_name, False)

    def get_signal_df(self) -> pd.DataFrame:
        if self.signal_observation_calendar is None:
            raise RuntimeError('Need to assign signal_observation_calendar before calculating signal')
        # first do the eligibility filter, then calculate the signal
        eligibility_df = self._get_eligibility_df()
        return self._calculate_signal(eligibility_df)

    def _get_eligibility_df(self) -> pd.DataFrame:
        if self.eligibility_df is None:
            # if eligibility DataFrame is not set, create a default one
            # default eligibility just checks if price has been published for the past 5 days
            start_date = self.signal_observation_calendar[0] - BDay(10)
            end_date = self.signal_observation_calendar[-1]
            underlying_price_df = self._financial_database_handler.get_close_price_df(self.ticker_list, start_date,
                                                                                      end_date)
            rolling_avg_df = underlying_price_df.rolling(window=5, min_periods=1).mean()
            eligibility_df = pd.DataFrame(data=np.where(rolling_avg_df.isna(), 0, 1),
                                          index=rolling_avg_df.index,
                                          columns=rolling_avg_df.columns)
            self.eligibility_df = select_rows_from_dataframe_based_on_sub_calendar(eligibility_df,
                                                                                   self.signal_observation_calendar)
        return self.eligibility_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter and static methods
    @staticmethod
    def _calculate_signal(eligibility_df: pd.DataFrame) -> pd.DataFrame:
        return eligibility_df  # to be overridden by subclasses

    @property
    def ticker_list(self):
        return self._ticker_list

    @property
    def signal_observation_calendar(self):
        return self._signal_observation_calendar

    @property
    def eligibility_df(self):
        return self._eligibility_df

    @ticker_list.setter
    def ticker_list(self, ticker_list: list):
        self._ticker_list = ticker_list
        self._eligibility_df = None  # reset the eligibility DataFrame

    @signal_observation_calendar.setter
    def signal_observation_calendar(self, signal_observation_calendar: pd.DatetimeIndex):
        self._signal_observation_calendar = signal_observation_calendar
        self._eligibility_df = None  # reset the eligibility DataFrame

    @eligibility_df.setter
    def eligibility_df(self, eligibility_df: pd.DataFrame):
        # check if eligibility_df has same tickers and observation dates and values are only 0s and 1s
        if eligibility_df is not None:
            # same column headers and allowed values are 1s and 0s
            # if signal_observation_calendar has already been assigned it needs to equal the index of eligibility_df
            change_eligibility_df = list(eligibility_df) == self.ticker_list \
                                    and check_if_values_in_dataframe_are_allowed(eligibility_df, 0, 1) \
                                    and all(eligibility_df.index == self.signal_observation_calendar) \
                                    if self.signal_observation_calendar is not None\
                                    else self.ticker_list \
                                    and check_if_values_in_dataframe_are_allowed(eligibility_df, 0, 1)
            if change_eligibility_df:
                self._eligibility_df = eligibility_df
                if self.signal_observation_calendar is None:
                    self._signal_observation_calendar = eligibility_df.index
            else:
                raise ValueError("eligibility_df does not have the correct format.")
        else:
            self._eligibility_df = eligibility_df


class _PriceBasedSignal(Signal):
    """Class definition of _PriceBasedSignal.
    Subclass of Signal. Overrides get_signal_df to check if price DataFrame has been assigned. Overrides the setters for
    ticker_list and signal_observation_calendar since they need to reset the price DataFrame."""

    def __init__(self, ticker_list: list, signal_observation_calendar: pd.DatetimeIndex,
                 eligibility_df: pd.DataFrame, total_return: bool, currency: str):
        Signal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df)
        self._total_return = total_return
        self._currency = currency
        self._underlying_price_df = None  # DataFrame is assigned using the method _set_underlying_price_df
        self._bday_before_start_date_buffer = 10  # add a some extra prices before start of signal observation calendar

    def _set_underlying_price_df(self):
        start_date = self.signal_observation_calendar[0] - BDay(self._bday_before_start_date_buffer)
        end_date = self.signal_observation_calendar[-1]
        if self.total_return:
            self._underlying_price_df = self._financial_database_handler.get_total_return_df(self.ticker_list,
                                                                                             start_date,
                                                                                             end_date,
                                                                                             currency=self.currency)
        else:
            self._underlying_price_df = self._financial_database_handler.get_close_price_df(self.ticker_list,
                                                                                            start_date,
                                                                                            end_date, self.currency)

    def get_signal_df(self) -> pd.DataFrame:
        if self.signal_observation_calendar is None:
            raise RuntimeError('Need to assign signal_observation_calendar before calculating signal')
        # assign a price DataFrame if applicable
        if self._underlying_price_df is None:
            self._set_underlying_price_df()
        eligibility_df = self._get_eligibility_df()  # then do the eligibility filter and calculate the signal
        return self._calculate_signal(eligibility_df)

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter
    @property
    def total_return(self):
        return self._total_return

    @total_return.setter
    def total_return(self, total_return: bool):
        self._total_return = total_return
        self._underlying_price_df = None  # reset the price DataFrame

    @property
    def currency(self):
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = currency

    @Signal.ticker_list.setter
    def ticker_list(self, ticker_list: list):
        self._ticker_list = ticker_list
        self._eligibility_df = None  # reset the eligibility DataFrame
        self._underlying_price_df = None  # reset the price DataFrame

    @Signal.signal_observation_calendar.setter
    def signal_observation_calendar(self, signal_observation_calendar: pd.DatetimeIndex):
        self._signal_observation_calendar = signal_observation_calendar
        self._eligibility_df = None  # reset the eligibility DataFrame
        self._underlying_price_df = None  # reset the price DataFrame


class _RankSignal(Signal):
    """Class definition of _RankSignal.
    Subclass of Signal."""

    def __init__(self, ticker_list: list, signal_observation_calendar: pd.DatetimeIndex, eligibility_df: pd.DataFrame,
                 rank_number: int, rank_fraction: float, descending: bool, include: bool):
        self._check_inputs(rank_number, rank_fraction)
        Signal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df)
        self.rank_number = rank_number
        self.rank_fraction = rank_fraction
        self.descending = descending
        self.include = include

    def _perform_ranking_on_dataframe(self, data_to_be_ranked: pd.DataFrame) -> pd.DataFrame:
        """Ranks data in a DataFrame in either descending or ascending order"""
        ranked_df = data_to_be_ranked.rank(axis='columns', method='first', ascending=not self.descending,
                                           numeric_only=True)
        if self.rank_number is not None:
            signal_array = np.where(ranked_df <= self.rank_number, self.include, not self.include)
        else:
            count_non_nan_s = ranked_df.count(axis=1)
            rank_number_s = round(count_non_nan_s * self.rank_fraction)
            signal_array = np.where(ranked_df.le(rank_number_s, axis=0), self.include,
                                    not self.include)  # True if df is Less or Equal to series
        signal_df = pd.DataFrame(index=data_to_be_ranked.index, columns=data_to_be_ranked.columns,
                                 data=signal_array)
        signal_df *= 1  # convert True to 1 and False to 0
        return signal_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter
    @staticmethod
    def _check_inputs(rank_number: int, rank_fraction: float):
        if rank_number is None and rank_fraction is None:
            raise ValueError('One needs to specify one of the two parameters rank_number and rank_fraction.')
        if rank_number is not None and rank_fraction is not None:
            raise ValueError('Of the two parameters rank_number and rank_fraction only one must be specified.')

    @property
    def rank_number(self):
        return self._rank_number

    @property
    def rank_fraction(self):
        return self._rank_fraction

    @rank_number.setter
    def rank_number(self, rank_number: int):
        if rank_number is None:
            self._rank_number = rank_number
        elif rank_number >= 1:
            self._rank_number = rank_number
            self._rank_fraction = None
        else:
            raise ValueError('rank_number needs to be greater or equal to 1.')

    @rank_fraction.setter
    def rank_fraction(self, rank_fraction: float):
        if rank_fraction is None:
            self._rank_fraction = rank_fraction
        elif 0.0 < rank_fraction < 1.0:
            self._rank_fraction = rank_fraction
            self._rank_number = None
        else:
            raise ValueError('rank_fraction needs to be greater than 0 and less than 1.')


class _PriceBasedRankSignal(_PriceBasedSignal, _RankSignal):
    """Class definition of _PriceBasedRankSignal.
    Subclass of _PriceBasedSignal and _RankSignal."""

    def __init__(self, ticker_list: list, signal_observation_calendar: pd.DatetimeIndex, eligibility_df: pd.DataFrame,
                 rank_number: int, rank_fraction: float, descending: bool, include: bool, total_return: bool,
                 currency: str):
        _RankSignal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df, rank_number, rank_fraction,
                             descending, include)
        _PriceBasedSignal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df, total_return,
                                   currency)


class VolatilityRankSignal(_PriceBasedRankSignal):
    """Class definition of VolatilityRankSignal.
    Subclass of _PriceBasedRankSignal."""

    def __init__(self, vol_lag: {int, tuple, list}, ticker_list: list = None, signal_observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, rank_number: int = None, rank_fraction: float = None,
                 descending: bool = False, include: bool = True, total_return: bool = True, currency: str = None):
        _PriceBasedRankSignal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df, rank_number,
                                       rank_fraction, descending, include, total_return, currency)
        self.vol_lag = vol_lag

    def _calculate_signal(self, eligibility_df: pd.DataFrame):
        """Calculates the realized volatility and then performs the ranking signal."""
        volatility_df = realized_volatility(self._underlying_price_df, *self.vol_lag)
        volatility_df = select_rows_from_dataframe_based_on_sub_calendar(volatility_df,
                                                                         eligibility_df.index)
        volatility_df *= eligibility_df.replace(0, np.nan)  # based on eligibility_df, if 0 replace volatility with nan.
        signal_df = self._perform_ranking_on_dataframe(volatility_df)
        return signal_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter
    @property
    def vol_lag(self):
        return self._vol_lag

    @vol_lag.setter
    def vol_lag(self, vol_lag):
        if isinstance(vol_lag, int):
            vol_lag = [vol_lag]
        if all(isinstance(lag, int) and lag > 1 for lag in vol_lag):  # all are int and larger than 1
            self._vol_lag = vol_lag
            self._bday_before_start_date_buffer = max(self._vol_lag) + 10
        else:
            raise ValueError('vol_lag needs to be an int or an iterable of int larger than 1')


class SimpleMovingAverageCrossSignal(_PriceBasedSignal):
    """Class definition of SimpleMovingAverageCrossSignal.
    Subclass of _PriceBasedSignal. Overrides the _calculate_signal method."""

    def __init__(self, leading_lagging_window: {list, tuple}, ticker_list: list = None,
                 signal_observation_calendar: pd.DatetimeIndex = None, eligibility_df: pd.DataFrame = None,
                 total_return: bool = False):
        _PriceBasedSignal.__init__(self, ticker_list, signal_observation_calendar, eligibility_df, total_return, None)
        self.leading_lagging_window = leading_lagging_window
        self._bday_before_start_date_buffer = self._lagging_window + 10

    def _calculate_signal(self, eligibility_df: pd.DataFrame):
        """Calculate the SMA crossover signal. Return a DataFrame.
        Given that eligibility_df shows 1, if SMA(lead) > SMA(lag) => 1 (bullish), else -1 (bearish) else 0."""
        # Calculate the SMA DataFrames and the signal before removal of NaN and eligibility constraints.
        leading_sma_df = rolling_average(self._underlying_price_df, self._leading_window)
        lagging_sma_df = rolling_average(self._underlying_price_df, self._lagging_window)
        signal_array = np.where(leading_sma_df > lagging_sma_df, 1, -1)
        sma_is_not_nan = ~(leading_sma_df + lagging_sma_df).isnull()
        signal_df = signal_array * sma_is_not_nan  # add 0 (neutral) if any SMA is NaN and apply the constraints
        # apply the eligibility filter
        signal_df = select_rows_from_dataframe_based_on_sub_calendar(signal_df, self.signal_observation_calendar)
        signal_df *= self.eligibility_df
        return signal_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter and static methods
    @property
    def leading_lagging_window(self):
        return self._leading_lagging_window

    @leading_lagging_window.setter
    def leading_lagging_window(self, leading_lagging_window: {list, tuple}):
        if min(leading_lagging_window) < 1:
            raise ValueError('Leading and lagging window need to be greater or equal to 1.')
        else:
            self._leading_window = int(min(leading_lagging_window))
            self._lagging_window = int(max(leading_lagging_window))
            self._leading_lagging_window = leading_lagging_window
            self._bday_before_start_date_buffer = self._lagging_window + 10

