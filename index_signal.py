import pandas as pd
from pandas.tseries.offsets import BDay
import pandas.core.common as com
import numpy as np
from financial_database import FinancialDatabase

# my own modules
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar, check_if_values_in_dataframe_are_allowed
from finance_tools import rolling_average


class Signal:
    """Class definition of Signal."""

    financial_database_name = r'sqlite:///C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\stock_db_v3'

    def __init__(self, ticker_list: list = None, signal_observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None):
        # either assign eligibility_df or ticker_list and signal_observation_calendar
        if eligibility_df is None and com.count_not_none(ticker_list, signal_observation_calendar) == 2:
            self._ticker_list = ticker_list
            self._signal_observation_calendar = signal_observation_calendar
        elif eligibility_df is not None and com.count_not_none(ticker_list, signal_observation_calendar) == 0:
            self._ticker_list = list(eligibility_df)
            self._signal_observation_calendar = eligibility_df.index
        else:
            raise ValueError("Need to assign ticker_list and signal_observation_calendar or only eligibility_df.")
        self._eligibility_df = eligibility_df
        self._financial_database_handler = FinancialDatabase(self.financial_database_name, False)

    def get_signal_df(self) -> pd.DataFrame:
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
        if eligibility_df is None \
                or (list(eligibility_df) == self.ticker_list
                    and all(eligibility_df.index == self.signal_observation_calendar)
                    and check_if_values_in_dataframe_are_allowed(eligibility_df, 0, 1)):
            self._eligibility_df = eligibility_df
        else:
            raise ValueError("eligibility_df does not have the correct format.")


class _PriceBasedSignal(Signal):
    """Class definition of _PriceBasedSignal.
    Subclass of Signal. Overrides get_signal_df to check if price DataFrame has been assigned. Overrides the setters for
    ticker_list and signal_observation_calendar since they need to reset the price DataFrame."""

    def __init__(self, ticker_list: list, signal_observation_calendar: pd.DatetimeIndex,
                 eligibility_df: pd.DataFrame, total_return: bool):
        super().__init__(ticker_list, signal_observation_calendar, eligibility_df)
        self._total_return = total_return  # since underlying_price_df will be set to None at init, no need to use setter
        self._underlying_price_df = None  # DataFrame is assigned using the method _set_underlying_price_df
        self._bday_before_start_date_buffer = 10  # add a some extra prices before start of signal observation calendar

    def _set_underlying_price_df(self):
        start_date = self.signal_observation_calendar[0] - BDay(self._bday_before_start_date_buffer)
        end_date = self.signal_observation_calendar[-1]
        if self.total_return:
            self._underlying_price_df = self._financial_database_handler.get_close_price_df(self.ticker_list,
                                                                                            start_date,
                                                                                            end_date)
        else:
            self._underlying_price_df = self._financial_database_handler.get_total_return_df(self.ticker_list,
                                                                                             start_date,
                                                                                             end_date)

    def get_signal_df(self) -> pd.DataFrame:
        # assign a price DataFrame if applicable
        if self._underlying_price_df is None:
            self._set_underlying_price_df()
        # then do the eligibility filter and calculate the signal
        eligibility_df = self._get_eligibility_df()
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


class SimpleMovingAverageCrossSignal(_PriceBasedSignal):
    """Class definition of SimpleMovingAverageCrossSignal.
    Subclass of _PriceBasedSignal. Overrides the _calculate_signal method."""

    def __init__(self, leading_lagging_window: {list, tuple}, ticker_list: list = None,
                 signal_observation_calendar: pd.DatetimeIndex = None, eligibility_df: pd.DataFrame = None,
                 total_return: bool = False):
        super().__init__(ticker_list, signal_observation_calendar, eligibility_df, total_return)
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

        # add 0 (neutral) if any SMA is NaN and apply the constraints
        signal_df = signal_array * sma_is_not_nan

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


def main():
    tickers = ["SAND.ST", "HM-B.ST", "AAK.ST"]
    dates = pd.date_range(start='2010', periods=50)
    main_signal = SimpleMovingAverageCrossSignal((1, 10), tickers, dates)
    sma = main_signal.get_signal_df()
    print(sma)


if __name__ == '__main__':
    main()





