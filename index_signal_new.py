"""
index_signal.py
"""
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas.core.common as com
import numpy as np


# my modules
from financial_database import FinancialDatabase
from config_database import my_database_name
from finance_tools import rolling_average
from dataframe_tools import merge_two_dataframes_as_of

# TODO test
from investment_universe import InvestmentUniverse


class Signal:
    """Class definition of Signal"""

    def __init__(self, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None):
        # if eligibility_df is not specified, create one from the tickers and observation calendar if they are both
        # available, else set it to None
        if eligibility_df is None:
            if com.count_not_none(tickers, observation_calendar) == 2:
                # reformat to list and capital letters
                if not isinstance(tickers, (list, tuple)):
                    tickers = [tickers.upper()]
                else:
                    tickers = [ticker.upper() for ticker in tickers]
                self.eligibility_df = pd.DataFrame(index=observation_calendar, columns=tickers, data=1)
            else:
                # set to None since either one of tickers or observation_calendar was not specified
                self._eligibility_df = None
        else:
            self.eligibility_df = eligibility_df

    def get_signal(self) -> pd.DataFrame:
        if self.eligibility_df is None:
            raise ValueError('An eligibility_df needs to be specified before calculating the signal.')
        else:
            raw_signal = self._calculate_signal()
            signal_at_obs_dates = merge_two_dataframes_as_of(pd.DataFrame(index=self.eligibility_df.index), raw_signal)
            clean_signal = signal_at_obs_dates * self.eligibility_df.values
            return clean_signal

    def _calculate_signal(self):
        # this method should be overridden when you want to change the signal
        return self.eligibility_df

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def eligibility_df(self):
        return self._eligibility_df

    @eligibility_df.setter
    def eligibility_df(self, eligibility_df: pd.DataFrame):
        if eligibility_df.index.is_monotonic_increasing and isinstance(eligibility_df.index, pd.DatetimeIndex):
            self._eligibility_df = eligibility_df
        else:
            raise ValueError('Index of eligibility_df needs to be a DatetimeIndex that is monotonic increasing.')

    def __repr__(self):
        return "<Signal()>"


class _FinancialDatabaseDependentSignal(Signal):
    """Class definition of _FinancialDatabaseDependentSignal. Subclass of Signal."""

    def __init__(self, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df)
        self._financial_database_handler = FinancialDatabase(my_database_name)

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def financial_database_handler(self):
        # make financial_database_handler read-only
        return self._financial_database_handler

    def __repr__(self):
        return "<_FinancialDatabaseDependentSignal()>"


class _PriceBasedSignal(_FinancialDatabaseDependentSignal):
    """Class definition of _PriceBasedSignal. Subclass of Signal."""

    def __init__(self, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, total_return: bool = False, currency: str = None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df)
        self.total_return = total_return
        self.currency = currency
        self.observation_buffer = 0

    def _get_start_end_date(self):
        return min(self.eligibility_df.index) - BDay(self.observation_buffer), max(self.eligibility_df.index)

    def _get_price_df(self):
        start_date, end_date = self._get_start_end_date()
        if self.total_return:
            return self.financial_database_handler.get_total_return_df(list(self.eligibility_df), start_date, end_date,
                                                                       0, self.currency)
        else:
            return self.financial_database_handler.get_close_price_df(list(self.eligibility_df), start_date, end_date,
                                                                      self.currency)

    def __repr__(self):
        return "<_PriceBasedSignal()>"


class SimpleMovingAverageCrossSignal(_PriceBasedSignal):
    """Class definition of SimpleMovingAverageCrossSignal. Subclass of _PriceBasedSignal.
    Given that eligibility_df is not nan, if SMA(lead) > SMA(lag) => 1 (bullish), else -1 (bearish) else 0."""
    def __init__(self, sma_lag_1: int, sma_lag_2: int = None, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, total_return: bool = False, currency: str = None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df,
                         total_return=total_return, currency=currency)
        self.sma_lag_1 = sma_lag_1
        self.sma_lag_2 = sma_lag_2
        self._sma_lead = min(self.sma_lag_1, self.sma_lag_2)
        self._sma_lag = max(self.sma_lag_1, self.sma_lag_2)
        self.observation_buffer = self._sma_lag + 10

    def _calculate_signal(self):
        price_data = self._get_price_df()
        leading_sma_df = rolling_average(price_data, self._sma_lead)
        lagging_sma_df = rolling_average(price_data, self._sma_lag)
        raw_signal_df = pd.DataFrame(index=price_data.index, columns=price_data.columns,
                                     data=np.where(leading_sma_df > lagging_sma_df, 1, -1))
        return raw_signal_df

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def sma_lag_1(self):
        return self._sma_lag_1

    @sma_lag_1.setter
    def sma_lag_1(self, sma_lag_1: int):
        if sma_lag_1 < 1:
            raise ValueError('sma_lag_1 needs to be an int larger or equal to 1.')
        else:
            self._sma_lag_1 = sma_lag_1

    @property
    def sma_lag_2(self):
        return self._sma_lag_2

    @sma_lag_2.setter
    def sma_lag_2(self, sma_lag_2: int):
        if sma_lag_2 is None:
            self._sma_lag_2 = 1
        elif sma_lag_2 < 1:
            raise ValueError('sma_lag_2 needs to be an int larger or equal to 1.')
        else:
            self._sma_lag_2 = sma_lag_2

    def __repr__(self):
        return "<SimpleMovingAverageCrossSignal()>"


def main():
    # AGRO.ST, AAK.ST, ABB.ST
    tickers = ["AGRO.ST", "AAK.ST", "ABB.ST"]
    invest_uni = InvestmentUniverse(tickers, '2018', '2020', freq='M')
    print(invest_uni)
    invest_uni.apply_liquidity_filter(60, 300000, 'sek')
    print(invest_uni)
    eligibility_df = invest_uni.get_eligibility_df(True)

    sma_signal = SimpleMovingAverageCrossSignal(50)
    sma_signal.eligibility_df = eligibility_df
    print(sma_signal.get_signal())


if __name__ == '__main__':
    main()
