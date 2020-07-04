"""
index_weight.py
"""
import pandas as pd
import numpy as np

import logging

# test
from investment_universe import InvestmentUniverse

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
    def __init__(self, signal_df: pd.DataFrame = None):
        self._signal_df = signal_df

    def get_weights(self):
        if self.signal_df is None:
            raise ValueError('signal_df not yet assigned.')
        return self._calculate_weight()

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

    def get_weight_desc(self):
        return 'no signal assigned' if self.signal_df is None else '{}x{} signal DataFrame assinged'.format(self.signal_df.shape[0], self.signal_df.shape[1])

    def __repr__(self):
        return "<Weight({})>".format(self.get_weight_desc())


class EqualWeight(Weight):
    """Class definition of EqualWeight. Subclass of Weight."""

    def _calculate_weight(self):
        # calculate the number of non-NaN columns per row
        numeric_columns = self.signal_df.count(axis=1)
        return self.signal_df.divide(numeric_columns, axis=0).replace(np.nan, 0)

    def __repr__(self):
        return "<EqualWeight({})>".format(self.get_weight_desc())


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

    def __repr__(self):
        return "<StaticWeight({})>".format(self.get_weight_desc())


def main():
    # AGRO.ST, AAK.ST, ABB.ST
    tickers = ["AGRO.ST", "AAK.ST", "ABB.ST"]
    invest_uni = InvestmentUniverse(tickers, '2018', '2020', freq='M')
    print(invest_uni)
    invest_uni.apply_liquidity_filter(60, 300000, 'sek')
    print(invest_uni)
    signal_df = invest_uni.get_eligibility_df(True)

    print(signal_df)
    eqw = StaticWeight()
    eqw.signal_df = signal_df
    print(eqw.get_weights())
    print(eqw)


if __name__ == '__main__':
    main()

