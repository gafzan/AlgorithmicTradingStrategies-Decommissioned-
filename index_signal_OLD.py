from financial_database_db_OLD import FinancialDataBase, underlying_attribute_dict, db_name  # for testing
import pandas as pd
import numpy as np
from finance_tools import rolling_average, realized_volatility
from dataframe_tools import dataframe_has_same_index_and_column_names, check_if_values_in_dataframe_are_allowed


class _Signal:
    """Class definition of _Signal."""

    def __init__(self, portfolio_eligibility_df: pd.DataFrame):
        self.portfolio_eligibility_df = portfolio_eligibility_df

    def get_signal(self) -> pd.DataFrame:
        """Returns a signal DataFrame after applying the eligibility filter."""
        signal_df = self._calculate_signal()
        return self._apply_portfolio_eligibility_filter(signal_df)

    def _calculate_signal(self):
        raise ValueError('Instance of a _Signal object should not calculate a signal. Only instances of subclasses '
                         'should.')

    def _apply_portfolio_eligibility_filter(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """If signal and eligibility DataFrames have the same format, multiply them and return the resulting
        DataFrame"""
        if self.portfolio_eligibility_df is not None:
            same_index_and_column = dataframe_has_same_index_and_column_names(signal_df, self.portfolio_eligibility_df)
            if same_index_and_column:
                signal_df *= self.portfolio_eligibility_df
                return signal_df
            else:
                raise ValueError('portfolio_eligibility_df does not have the same composition as signal_df.')
        else:
            return signal_df  # return signal without any extra filters

    @property
    def portfolio_eligibility_df(self) -> pd.DataFrame:
        return self._portfolio_eligibility_df

    @portfolio_eligibility_df.setter
    def portfolio_eligibility_df(self, portfolio_eligibility_df: pd.DataFrame):
        if portfolio_eligibility_df is None or check_if_values_in_dataframe_are_allowed(portfolio_eligibility_df, 0, 1):
            self._portfolio_eligibility_df = portfolio_eligibility_df
        else:
            raise ValueError('portfolio_eligibility_df contains values different from 0 and 1.')


class _PriceBasedSignal(_Signal):
    """Class definition of _PriceBasedSignal."""

    def __init__(self, underlying_price_df: pd.DataFrame, portfolio_eligibility_df: pd.DataFrame,
                 total_return_flag: bool):
        _Signal.__init__(self, portfolio_eligibility_df)
        self.underlying_price_df = underlying_price_df
        self.portfolio_eligibility_df = portfolio_eligibility_df
        self.total_return_flag = total_return_flag

    @property
    def underlying_price_df(self):
        return self._underlying_price_df

    @underlying_price_df.setter
    def underlying_price_df(self, underlying_price_df: pd.DataFrame):
        self._underlying_price_df = underlying_price_df
        self.portfolio_eligibility_df = None


class _RankSignal(_Signal):
    """Class definition of _RankSignal"""

    def __init__(self, rank_number: int, rank_fraction: float,
                 descending: bool, include: bool, portfolio_eligibility_df: pd.DataFrame):
        self._check_inputs(rank_number, rank_fraction)
        _Signal.__init__(self, portfolio_eligibility_df)
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
            signal_array = np.where(ranked_df.le(rank_number_s, axis=0), self.include, not self.include)  # True if df is Less or Equal to series
        signal_df = pd.DataFrame(index=data_to_be_ranked.index, columns=data_to_be_ranked.columns,
                                 data=signal_array)
        return signal_df

    def _calculate_dataframe_to_be_ranked(self):
        # this is to be overridden by subclasses
        raise TypeError("_calculate_dataframe_to_be_ranked should not be called by an instance of a _RankSignal object.")

    @property
    def rank_number(self):
        return self._rank_number

    @property
    def rank_fraction(self):
        return self._rank_fraction

    @rank_number.setter
    def rank_number(self, rank_number: int):
        if rank_number is None or rank_number >= 1:
            self._rank_number = rank_number
            self._rank_fraction = None
        else:
            raise ValueError('rank_number needs to be greater or equal to 1.')

    @rank_fraction.setter
    def rank_fraction(self, rank_fraction: float):
        if rank_fraction is None or 0.0 < rank_fraction < 1.0:
            self._rank_fraction = rank_fraction
            self._rank_number = None
        else:
            raise ValueError('rank_fraction needs to be greater than 0 and less than 1.')

    @staticmethod
    def _check_inputs(rank_number: int, rank_fraction: float):
        if rank_number is None and rank_fraction is None:
            raise ValueError('One needs to specify one of the two parameters rank_number and rank_fraction.')
        if rank_number is not None and rank_fraction is not None:
            raise ValueError('Of the two parameters rank_number and rank_fraction only one must be specified.')
    pass


class SimpleMovingAverageCross(_PriceBasedSignal):
    """Class definition of SimpleMovingAverageCross."""

    def __init__(self, leading_window: int, lagging_window: int, underlying_price_df: pd.DataFrame = None,
                 portfolio_eligibility_df: pd.DataFrame = None):
        self._check_inputs(leading_window, lagging_window)
        _PriceBasedSignal.__init__(self, underlying_price_df, portfolio_eligibility_df, False)
        self.leading_window = leading_window
        self.lagging_window = lagging_window

    @staticmethod
    def _check_inputs(leading_window: int, lagging_window: int):
        if leading_window < 1 or lagging_window < 1:
            raise ValueError('leading_window and lagging_window needs to be an integer larger or equal to 1.')
        if leading_window > lagging_window:
            raise ValueError('leading_window needs to be less than or equal to lagging_window')

    def _calculate_signal(self):
        """Calculate the SMA crossover signal. Return a DataFrame. If SMA(lead) > SMA(lag) => 1 (bullish),
        else -1 (bearish)."""
        if self.underlying_price_df is None:
            raise ValueError('underlying_price_df has not yet been assigned.')
        # Calculate the SMA DataFrames and the signal before removal of NaN and eligibility constraints.
        leading_sma_df = rolling_average(self.underlying_price_df, self.leading_window)
        lagging_sma_df = rolling_average(self.underlying_price_df, self.lagging_window)
        signal_df = np.where(leading_sma_df > lagging_sma_df, 1, -1)
        sma_is_not_nan = ~(leading_sma_df + lagging_sma_df).isnull() * 1

        # Add 0 (neutral) if any SMA is NaN and apply the constraints
        signal_df *= sma_is_not_nan
        # self._apply_portfolio_eligibility_filter(signal_df)
        return signal_df

    def __repr__(self):
        if self.portfolio_eligibility_df is None:
            constraints = 'no'
        else:
            constraints = 'with'
        return "<SimpleMovingAverageCross(leading window = {}, lagging window = {}, {} constraints)>"\
            .format(self.leading_window, self.lagging_window, constraints)


class _PriceBasedRankSignal(_PriceBasedSignal, _RankSignal):
    """Class definition of _PriceBasedRanksSignal"""
    def __init__(self, underlying_price_df: pd.DataFrame, rank_number: int, rank_fraction: float,
                 descending: bool, include: bool, portfolio_eligibility_df: pd.DataFrame,
                 total_return_flag: bool):
        _PriceBasedSignal.__init__(self, underlying_price_df, portfolio_eligibility_df, total_return_flag)
        _RankSignal.__init__(self, rank_number, rank_fraction, descending, include, portfolio_eligibility_df)

    def _calculate_signal(self):
        data_to_be_ranked_df = self._calculate_dataframe_to_be_ranked()
        return self._perform_ranking_on_dataframe(data_to_be_ranked_df)


class VolatilityRankSignal(_PriceBasedRankSignal):
    """Class definition of VolatilityRankSignal"""

    def __init__(self, volatility_lag: {int, list}, underlying_price_df: pd.DataFrame = None, rank_number: int = None,
                 rank_fraction: float = None, descending: bool = False, include: bool = True,
                 portfolio_eligibility_df: pd.DataFrame = None, total_return_flag: bool = False):
        _PriceBasedRankSignal.__init__(self, underlying_price_df, rank_number, rank_fraction, descending, include,
                                       portfolio_eligibility_df, total_return_flag)
        self.volatility_lag = volatility_lag

    @property
    def volatility_lag(self):
        return self._volatility_lag

    @volatility_lag.setter
    def volatility_lag(self, volatility_lag: {int, list}):
        if isinstance(volatility_lag, int):
            if volatility_lag < 2:
                raise TypeError('volatility_lag needs to be greater or equal to 1. (current value = {})'
                                .format(volatility_lag))
        elif isinstance(volatility_lag, list):
            for lag in volatility_lag:
                if lag < 2:
                    raise TypeError('volatility_lag needs to be greater or equal to 1.')
        else:
            raise TypeError('volatility_lag needs to be of type int or list.')
        self._volatility_lag = volatility_lag

    def _calculate_dataframe_to_be_ranked(self):
        if isinstance(self.volatility_lag, int):
            volatility_df = realized_volatility(self.underlying_price_df, vol_lag=self.volatility_lag)
        else:
            # when volatility_lag is a list, take the maximum volatility
            max_volatility_df = None
            counter = 0
            for lag in self.volatility_lag:
                volatility_sub_df = realized_volatility(self.underlying_price_df, vol_lag=lag)
                if counter == 0:
                    max_volatility_df = volatility_sub_df
                else:
                    max_volatility_df = pd.concat([max_volatility_df, volatility_sub_df]).max(level=0, skipna=False)
                counter += 1
            volatility_df = max_volatility_df
        return volatility_df

    def __repr__(self):
        if self.include:
            rank_desc_str = 'pick the '
        else:
            rank_desc_str = 'remove the '
        if self.descending:
            rank_desc_str += 'top '
        else:
            rank_desc_str += 'bottom '
        if self.rank_number is not None:
            rank_desc_str += str(self.rank_number)
        else:
            rank_desc_str += str(round(self.rank_fraction * 100, 2)) + '%'
        return '<VolatilityRankSignal(volatility lag = {}, {}>'.format(self.volatility_lag, rank_desc_str)


def is_signal_subclass_instance(obj):
    return issubclass(type(obj), _Signal)


def is_price_based_signal_subclass_instance(obj):
    return issubclass(type(obj), _PriceBasedSignal)


def main():
    # get the data
    fin_db = FinancialDataBase(db_name, db_echo=False)
    attribute_dict = underlying_attribute_dict(currency='SEK', industry='BIOTECHNOLOGY')
    tickers = fin_db.get_ticker(attribute_dict)
    tickers = tickers[: 3]
    price_df = fin_db.get_close_price_df(tickers, start_date='2018-01-01')

    # liquidity constraints
    liquidity_df = fin_db.get_liquidity_df(tickers, start_date='2018-01-01')
    avg_liquidity_df = rolling_average(liquidity_df, 60).fillna(0)
    liquidity_threshold = 400000
    eligable_for_investment = pd.DataFrame(np.where(avg_liquidity_df.values > liquidity_threshold, 1, 0),
                                           index=avg_liquidity_df.index,
                                           columns=avg_liquidity_df.columns)
    #
    # # set up the strategy
    # lead = 5
    # lag = 10
    # main_sma_signal = SimpleMovingAverageCross(lead, lag)
    # print(main_sma_signal)
    #
    # main_sma_signal.portfolio_eligibility_df = eligable_for_investment
    # print(main_sma_signal)

    # ranking strategy
    vol_rank_signal = VolatilityRankSignal(volatility_lag=[5, 10], rank_fraction=0.5)
    vol_rank_signal.underlying_price_df = price_df
    print(vol_rank_signal.get_signal())
    vol_rank_signal.portfolio_eligibility_df = eligable_for_investment
    print(vol_rank_signal.get_signal())
    print(vol_rank_signal)


if __name__ == '__main__':
    main()
