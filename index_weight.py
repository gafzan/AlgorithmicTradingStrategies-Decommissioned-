import pandas as pd
import numpy as np
from financial_database import FinancialDatabase
from config_database import my_database_name
from dataframe_tools import dataframe_has_same_index_and_column_names, check_if_values_in_dataframe_are_allowed, \
    select_rows_from_dataframe_based_on_sub_calendar, get_unique_values_from_dataframe
from finance_tools import realized_volatility
from datetime import timedelta
from index_signal import SimpleMovingAverageCrossSignal


class _Weight:
    """Class definition of Weight"""

    # Initializer / instance attribute
    def __init__(self, signal_df: pd.DataFrame = None):
        self.signal_df = signal_df

    def get_weights(self):
        if self.signal_df is None:
            raise ValueError('signal_df not yet assigned.')
        return self._calculate_weight()

    def _calculate_weight(self):
        # to be overridden
        raise ValueError('Instance of a _Weight object should not call calculate_weight.')

    @property
    def signal_df(self):
        return self._signal_df

    @signal_df.setter
    def signal_df(self, signal_df: pd.DataFrame):
        if signal_df is None or check_if_values_in_dataframe_are_allowed(signal_df, -1, 0, 1):
            self._signal_df = signal_df
        else:
            raise ValueError('Signal DataFrame contains more values than -1, 0 and 1.')

    def __repr__(self):
        return "<Weight()>"


class EqualWeight(_Weight):
    """Class definition of EqualWeight."""

    def __init__(self, total_long_allocation: float = 1.0, total_short_allocation: float = 0,
                 signal_df: pd.DataFrame = None):
        super().__init__(signal_df)
        self.total_long_allocation = total_long_allocation
        self.total_short_allocation = total_short_allocation

    def _calculate_weight(self):
        return self._get_equal_weights(self.signal_df, self.total_long_allocation, self.total_short_allocation)

    @staticmethod
    def _get_equal_weights(signal_df: pd.DataFrame, total_long_allocation: float = 1.0,
                           total_short_allocation: float = 0.0) -> pd.DataFrame:
        """Assumes that signal_df is a DataFrame containing 0 (neutral), 1 (long) or -1 (short) and that
        maximum_long_allocation maximum_short_allocation are float."""
        shifted_signal_df = signal_df.shift()  # Observe signal at T and adjust weights at T + 1
        shifted_signal_df.iloc[0, :] = 0.0  # Set the first row (NaN) to zero.

        count_long_positions_s = (shifted_signal_df == 1.0).sum(axis=1)  # Sum each row
        count_short_positions_s = (shifted_signal_df == -1.0).sum(axis=1)

        # Calculate the weights and remove any non-numeric values
        long_weights_s = total_long_allocation / count_long_positions_s
        short_weights_s = -1 * total_short_allocation / count_short_positions_s
        long_weights_s.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)
        short_weights_s.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        shifted_long_signal_df = shifted_signal_df.replace(-1.0, 0.0)  # Only include the long and neutral signal
        shifted_short_signal_df = shifted_signal_df.replace(1.0, 0.0)  # Only include the short and neutral signal
        long_weights_df = shifted_long_signal_df.mul(long_weights_s, axis=0)
        short_weights_df = shifted_short_signal_df.mul(short_weights_s, axis=0)
        weights_df = long_weights_df + short_weights_df
        return weights_df

    def __repr__(self):
        return "<EqualWeight(total 'bullish' allocation = {}%, total 'bearish' allocation = {}%)>"\
            .format(round(self.total_long_allocation * 100, 2), round(self.total_short_allocation * 100, 2))


class StaticWeight(_Weight):
    """Class definition of StaticWeight"""

    def _calculate_weight(self):
        return self._get_static_weights(self.signal_df)

    # Overriding the setter property in the parent class because of different constraint on signal DataFrame.
    @_Weight.signal_df.setter
    def signal_df(self, signal_df: pd.DataFrame):
        self._signal_df = signal_df

    @staticmethod
    def _get_static_weights(signal_df: pd.DataFrame) -> pd.DataFrame:
        """Assumes that signal_df is a DataFrame containing 0 (neutral) or 1 (long). Returns a DataFrame with the
        static weights chosen by the user."""
        counter = 0
        weight_df = pd.DataFrame(index=signal_df.index, columns=signal_df.columns)
        for ticker in list(signal_df):
            signal_values = set(signal_df.loc[:, ticker].values)  # unique signal values for each ticker
            try:
                signal_values.remove(0)
            except KeyError:
                pass
            # ask user to insert each weight corresponding to the values of the signal (except 0) for each ticker
            signal_weight_dict = {}
            for signal in signal_values:
                ask_user = True
                while ask_user:
                    try:
                        weight_from_user = float(input(f'Enter weight(%) for {ticker} ({counter + 1}/{len(list(signal_df))}) '
                                                       f'during signal = {signal} {signal_values}: '))
                    except ValueError:
                        pass
                    else:
                        signal_weight_dict.update({signal: weight_from_user / 100})
                        ask_user = False
            weight_df[ticker] = signal_df[ticker].map(signal_weight_dict)
            counter += 1
        return weight_df


class _ProportionalValueWeight(_Weight):
    """Class definition for _ProportionalValueWeight. Subclass of Weight class. Weights are proportional to values
    from a database."""

    def __init__(self, signal_df: pd.DataFrame, inversely: bool, measurement_lag: int):
        super().__init__(signal_df)
        self.inversely = inversely
        self.measurement_lag = measurement_lag
        self._financial_database_handler = FinancialDatabase(my_database_name, database_echo=False)

    def _calculate_weight(self):
        values_df = self._get_values_for_weight_calculation_df()
        return self._get_proportional_weights(self.signal_df, values_df, self.inversely)

    def _get_values_for_weight_calculation_df(self) -> pd.DataFrame:
        raise ValueError('_get_values_for_weight_calculation_df should not be called by an instance of a '
                         '_ProportionalValueWeight object.')

    def _get_tickers_start_end_date(self):
        tickers = list(self.signal_df.columns)
        start_date = self.signal_df.index[0]
        end_date = self.signal_df.index[-1]
        return tickers, start_date, end_date

    @staticmethod
    def _get_proportional_weights(signal_df: pd.DataFrame, values_df: pd.DataFrame, inversely: bool) -> pd.DataFrame:
        """Assumes signal_df and values_df are two DataFrames with the same index and columns. inversely is bool and
        decides if the weights are proportional or inversely-proportional."""
        if not dataframe_has_same_index_and_column_names(signal_df, values_df):
            raise ValueError('signal_df and values_df does not have the same composition.')
        values_df.iloc[:, 0] = np.nan
        if inversely:
            values_df = values_df.apply(lambda x: 1.0 / x)
        values_df *= signal_df
        values_sum_s = values_df.sum(axis=1)
        proportional_weight_df = values_df.div(values_sum_s, axis='index').fillna(value=0)
        return proportional_weight_df

    # Overriding the setter property in the parent class because of different constraint on signal DataFrame.
    @_Weight.signal_df.setter
    def signal_df(self, signal_df: pd.DataFrame):
        if signal_df is None or check_if_values_in_dataframe_are_allowed(signal_df, 0, 1):
            self._signal_df = signal_df
        else:
            raise ValueError('Signal DataFrame contains more values than 0 and 1.')

    @property
    def measurement_lag(self):
        return self._measurement_lag

    @measurement_lag.setter
    def measurement_lag(self, measurement_lag: int):
        if measurement_lag >= 1:
            self._measurement_lag = measurement_lag
        else:
            raise ValueError('Measurement lag needs to be greater or equal to 1.')


class VolatilityWeight(_ProportionalValueWeight):
    """Class definition of VolatilityWeight"""

    def __init__(self, volatility_lag: int, inversely: bool = True, signal_df: pd.DataFrame = None,
                 total_return: bool = False):
        super().__init__(signal_df, inversely, volatility_lag)
        self.volatility_lag = volatility_lag
        self.total_return = total_return

    def _get_values_for_weight_calculation_df(self):
        tickers, start_date, end_date = self._get_tickers_start_end_date()
        start_date -= timedelta(days=self.volatility_lag + 20)  # add some extra dates
        if self.total_return:
            price_df = self._financial_database_handler.get_total_return_df(tickers, start_date, end_date)
        else:
            price_df = self._financial_database_handler.get_close_price_df(tickers, start_date, end_date)
        # calculate the volatility and match the index with the signal DataFrame
        volatility_df = select_rows_from_dataframe_based_on_sub_calendar(
            realized_volatility(price_df, self.volatility_lag),
            self.signal_df.index)
        return volatility_df

    def __repr__(self):
        return "<VolatilityWeight(inversely = {}, measurement lag = {} days)>".format(self.inversely, self.volatility_lag)
