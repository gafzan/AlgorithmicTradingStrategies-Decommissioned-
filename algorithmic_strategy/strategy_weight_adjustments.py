"""
strategy_weight_adjustments.py
"""

import pandas as pd
import numpy as np
from financial_analysis.finance_tools import realized_volatility


class _WeightAdjustment:
    """Class definition of _WeightAdjustment"""

    def __init__(self, lag: int):
        self.lag = lag

    def perform_adjustment(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd):
        return self._adjustment_method(multivariate_daily_return_df, multivariate_daily_weight_df)

    def _adjustment_method(self, multivariate_daily_return_df: pd.DataFrame, multivariate_daily_weight_df: pd.DataFrame)\
            ->(pd.DataFrame, pd.DataFrame):
        raise ValueError('_adjustment_method is not defined')

    def _calculate_strategy_return(self, multivariate_daily_return_df: pd.DataFrame,
                                   multivariate_daily_weight_df: pd.DataFrame, sum_columns: bool = True) -> pd.DataFrame:
        strategy_df = multivariate_daily_return_df * multivariate_daily_weight_df.shift(self.lag)
        if sum_columns:
            strategy_df = pd.DataFrame({'strategy_return': strategy_df.sum(axis=1, skipna=False).values})
        return strategy_df

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods
    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, lag: int):
        if lag >= 1:
            self._lag = lag
        else:
            raise ValueError('lag needs to be greater or equal to one in order to make the strategy replicable')

    @staticmethod
    def _get_desc():
        return 'none'

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self._get_desc())


class VolatilityControl(_WeightAdjustment):
    """Class definition of VolatilityControl"""

    def __init__(self, vol_control_level: float, vol_lag: {int, list, tuple}, max_exposure: float = 1.0, lag: int = 2):
        super().__init__(lag)
        self.vol_control_level = vol_control_level
        self.vol_lag = vol_lag
        self.max_exposure = max_exposure

    def _adjustment_method(self, multivariate_daily_return_df, multivariate_daily_weight_df):
        strategy_return_df = self._calculate_strategy_return(multivariate_daily_return_df, multivariate_daily_weight_df)
        realized_vol = realized_volatility(multivariate_return_df=strategy_return_df, vol_lag=self.vol_lag)
        constraint_func = lambda vol: self.vol_control_level / vol if self.vol_control_level / vol < self.max_exposure \
            else (vol if np.isnan(vol) else self.max_exposure)
        scaling_factor = realized_vol[list(realized_vol)[0]].apply(constraint_func)

        return multivariate_daily_weight_df, multivariate_daily_weight_df
        # handle the constraints and nan


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
    def max_exposure(self):
        return self._max_exposure

    @max_exposure.setter
    def max_exposure(self, max_exposure: float):
        if max_exposure > 0:
            self._max_exposure = max_exposure
        else:
            raise ValueError('max_exposure needs to be strictly larger than 0')


def main():
    adj = _WeightAdjustment
    adj.perform_adjustment()
    print(adj)


if __name__ == '__main__':
    main()
