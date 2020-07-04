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
from finance_tools import rolling_average, realized_volatility_v2
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
            if list(raw_signal) == list(self.eligibility_df):
                signal_at_obs_dates = merge_two_dataframes_as_of(pd.DataFrame(index=self.eligibility_df.index), raw_signal)
            else:
                raise ValueError('column headers of the eligibility_df and the ones from the raw_signal DataFrame are not the same.')
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


class _RankSignal(Signal):
    """Class definition of _RankSignal.
    Subclass of Signal."""

    def __init__(self, tickers: {str, list}, observation_calendar: pd.DatetimeIndex, eligibility_df: pd.DataFrame,
                 rank_number: int, rank_fraction: float, descending: bool, include: bool):
        Signal.__init__(self, tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df)
        if com.count_not_none(rank_number, rank_fraction) == 2:
            raise ValueError('Of the two parameters: rank_number and rank_fraction only one can be specified.')
        self.rank_number = rank_number
        self.rank_fraction = rank_fraction
        self.descending = descending
        self.include = include

    def _perform_ranking_on_dataframe(self, data_to_be_ranked: pd.DataFrame) -> pd.DataFrame:
        """Ranks data in a DataFrame in either descending or ascending order"""
        if com.count_not_none(self.rank_number, self.rank_fraction) == 0:
            raise ValueError('Need to secify rank_number or rank_fraction before performing ranking.')
        # ignore instruments that are not eligible
        eligible_df = merge_two_dataframes_as_of(pd.DataFrame(index=data_to_be_ranked.index), self.eligibility_df)
        data_to_be_ranked_eligible = data_to_be_ranked * eligible_df.values
        ranked_df = data_to_be_ranked_eligible.rank(axis='columns', method='first', ascending=not self.descending,
                                                    numeric_only=True)
        if self.rank_fraction is None:
            signal_array = np.where(ranked_df <= self.rank_number, self.include, not self.include)
        else:
            count_non_nan_s = ranked_df.count(axis=1)
            rank_number_s = round(count_non_nan_s * self.rank_fraction)
            signal_array = np.where(ranked_df.le(rank_number_s, axis=0), self.include,
                                    not self.include)  # True if df is Less or Equal to series
        signal_df = pd.DataFrame(index=data_to_be_ranked_eligible.index, columns=data_to_be_ranked_eligible.columns,
                                 data=signal_array)
        signal_df *= 1  # convert True to 1 and False to 0
        return signal_df

    def _calculate_signal(self):
        data = self._get_dataframe_to_be_ranked()
        return self._perform_ranking_on_dataframe(data)

    def _get_dataframe_to_be_ranked(self):
        raise ValueError('_get_dataframe_to_be_ranked should not be called by an instance of _RankSignal.')

    # ------------------------------------------------------------------------------------------------------------------
    # getter, setter
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
            self.rank_fraction = None
        else:
            raise ValueError('rank_number needs to be an int greater or equal to 1.')

    @rank_fraction.setter
    def rank_fraction(self, rank_fraction: float):
        if rank_fraction is None:
            self._rank_fraction = rank_fraction
        elif 0.0 < rank_fraction < 1.0:
            self._rank_fraction = rank_fraction
            self.rank_number = None
        else:
            raise ValueError('rank_fraction needs to be greater than 0 and less than 1.')


class _FinancialDatabaseDependentSignal(Signal):
    """Class definition of _FinancialDatabaseDependentSignal. Subclass of Signal."""

    def __init__(self, tickers: {str, list}, observation_calendar: pd.DatetimeIndex,
                 eligibility_df: pd.DataFrame):
        Signal.__init__(self, tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df)
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

    def __init__(self, tickers: {str, list}, observation_calendar: pd.DatetimeIndex,
                 eligibility_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int}):
        _FinancialDatabaseDependentSignal.__init__(self, tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df)
        self.total_return = total_return
        self.currency = currency
        self._observation_buffer = 0
        self._weekday_i_dict = {'mon': 0, 'monday': 0, 'tuesday': 1, 'tue': 1, 'wed': 2, 'wednesday': 2, 'thursday': 3,
                                'thu': 4, 'friday': 5, 'fri': 5}
        self.price_obs_freq = price_obs_freq

    def _get_start_end_date(self):
        if self.eligibility_df is None:
            raise ValueError('An eligibility_df needs to be specified.')
        return min(self.eligibility_df.index) - BDay(self._observation_buffer), max(self.eligibility_df.index)

    def _get_price_df(self):
        start_date, end_date = self._get_start_end_date()
        if self.total_return:
            price = self.financial_database_handler.get_total_return_df(list(self.eligibility_df), start_date, end_date,
                                                                        0, self.currency)
        else:
            price = self.financial_database_handler.get_close_price_df(list(self.eligibility_df), start_date, end_date,
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

    def __repr__(self):
        return "<_PriceBasedSignal()>"


class _PriceBasedRankSignal(_PriceBasedSignal, _RankSignal):
    """Class definition of _PriceBasedRankSignal. Subclass of _PriceBasedSignal and _RankSignal."""

    def __init__(self, tickers: {str, list}, observation_calendar: pd.DatetimeIndex,
                 eligibility_df: pd.DataFrame, total_return: bool, currency: str, price_obs_freq: {str, int},
                 rank_number: int, rank_fraction: float, descending: bool, include: bool):
        _PriceBasedSignal.__init__(self, tickers=tickers, observation_calendar=observation_calendar,
                                   eligibility_df=eligibility_df, total_return=total_return, currency=currency,
                                   price_obs_freq=price_obs_freq)
        _RankSignal.__init__(self, tickers=tickers, observation_calendar=observation_calendar,
                             eligibility_df=eligibility_df, rank_number=rank_number, rank_fraction=rank_fraction,
                             descending=descending, include=include)

    def __repr__(self):
        return "<_PriceBasedRankSignal()>"


class SimpleMovingAverageCrossSignal(_PriceBasedSignal):
    """Class definition of SimpleMovingAverageCrossSignal. Subclass of _PriceBasedSignal.
    Given that eligibility_df is not nan, if SMA(lead) > SMA(lag) => 1 (bullish), else -1 (bearish) else 0."""
    def __init__(self, sma_lag_1: int, sma_lag_2: int = None, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, total_return: bool = False, currency: str = None, price_obs_freq: {str, int}=None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df,
                         total_return=total_return, currency=currency, price_obs_freq=price_obs_freq)
        self.sma_lag_1 = sma_lag_1
        self.sma_lag_2 = sma_lag_2
        self._sma_lead = min(self.sma_lag_1, self.sma_lag_2)
        self._sma_lag = max(self.sma_lag_1, self.sma_lag_2)
        self._observation_buffer = self._sma_lag + 10

    def _calculate_signal(self):
        price_data = self._get_price_df()
        leading_sma_df = rolling_average(price_data, self._sma_lead)
        lagging_sma_df = rolling_average(price_data, self._sma_lag)
        raw_signal_df = pd.DataFrame(index=price_data.index, columns=price_data.columns,
                                     data=np.where(leading_sma_df < lagging_sma_df, -1, 1))
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


class VolatilityRankSignal(_PriceBasedRankSignal):
    """Class definition of VolatilityRankSignal. Subclass of _PriceBasedRankSignal."""
    def __init__(self, volatility_observation_period: {int, list}, rank_number: int = None, rank_fraction: float = None,
                 descending: bool = True, include: bool = True, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, total_return: bool = None, currency: str = None, price_obs_freq: {str, int} = None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df,
                         total_return=total_return, currency=currency, price_obs_freq=price_obs_freq,
                         rank_number=rank_number, rank_fraction=rank_fraction, descending=descending, include=include)
        self.volatility_observation_period = volatility_observation_period
        if isinstance(volatility_observation_period, int):
            self._observation_buffer = volatility_observation_period + 10
        else:
            self._observation_buffer = max(volatility_observation_period) + 10

    def _get_dataframe_to_be_ranked(self):
        price = self._get_price_df()
        volatility = realized_volatility_v2(multivariate_price_df=price, vol_lag=self.volatility_observation_period)
        return volatility

    def __repr__(self):
        return "<VolatilityRankSignal()>"


class PerformanceRankSignal(_PriceBasedRankSignal):
    """Class definition of VolatilityRankSignal. Subclass of _PriceBasedRankSignal."""
    def __init__(self, performance_observation_period: int, rank_number: int = None, rank_fraction: float = None,
                 descending: bool = True, include: bool = True, tickers: {str, list}=None, observation_calendar: pd.DatetimeIndex = None,
                 eligibility_df: pd.DataFrame = None, total_return: bool = None, currency: str = None, price_obs_freq: {str, int}=None):
        super().__init__(tickers=tickers, observation_calendar=observation_calendar, eligibility_df=eligibility_df,
                         total_return=total_return, currency=currency, price_obs_freq=price_obs_freq,
                         rank_number=rank_number, rank_fraction=rank_fraction, descending=descending, include=include)
        self.performance_observation_period = performance_observation_period
        self._observation_buffer = performance_observation_period + 10

    def _get_dataframe_to_be_ranked(self):
        price = self._get_price_df()
        performance = price.pct_change(self.performance_observation_period)
        return performance

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def performance_observation_period(self):
        return self._performance_observation_period

    @performance_observation_period.setter
    def performance_observation_period(self, performance_observation_period: int):
        if performance_observation_period >= 1:
            self._performance_observation_period = performance_observation_period
        else:
            raise ValueError('performance_observation_period needs to be an integer larger or equal to 1.')

    def __repr__(self):
        return "<PerformanceRankSignal()>"


def main():
    # AGRO.ST, AAK.ST, ABB.ST
    tickers = ["AGRO.ST", "AAK.ST", "ABB.ST"]
    invest_uni = InvestmentUniverse(tickers, '2018', '2020', freq='M')
    invest_uni.apply_liquidity_filter(60, 300000)
    eligibility_df = invest_uni.get_eligibility_df(True)

    low_vol_signal = VolatilityRankSignal(60, 1, descending=True)
    low_vol_signal.eligibility_df = eligibility_df
    print(low_vol_signal.get_signal())

    # sma_signal = SimpleMovingAverageCrossSignal(100, 20, eligibility_df=eligibility_df)
    # print(sma_signal.get_signal())




if __name__ == '__main__':
    main()
