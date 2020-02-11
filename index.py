from financial_database import FinancialDatabase, Underlying
from config_database import my_database_name
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar
from finance_tools import index_calculation, rolling_average
from index_signal_OLD import is_signal_subclass_instance, is_price_based_signal_subclass_instance, SimpleMovingAverageCross
from index_weight import is_weight_subclass_instance, EqualWeight
import logging

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class Index:
    """Class definition for Index."""

    _price_availability_observation_threshold = 5  # when no signal is assigned, a default signal equals 1 if there is a
    # price in the past '_price_availability_observation_threshold' days

    # Initializer / instance attributes
    def __init__(self, rebalancing_calendar: pd.DatetimeIndex, tickers: list,
                 end_date: datetime = date.today(),
                 fee: float = 0.0,
                 transaction_cost: float = 0.0,
                 initial_amount: float = 100.0,
                 total_return: bool = False,
                 tax_on_dividend: float = 0.0):
        logger.debug('Initializing an Index object')
        if not rebalancing_calendar.is_monotonic_increasing:
            raise ValueError('The rebalancing calendar needs to be monotonic increasing. '
                             'Use rebalancing_calendar.sort_values().')
        if end_date < rebalancing_calendar[-1]:
            raise ValueError('end_date occurs before the end of the rebalance calendar.')
        self._tickers = tickers
        self._end_date = end_date
        self._rebalancing_calendar = rebalancing_calendar
        self.total_return = total_return
        self.tax_on_dividend = tax_on_dividend
        self.fee = fee
        self.transaction_cost = transaction_cost
        self.initial_amount = initial_amount
        self._financial_database = FinancialDatabase(my_database_name, database_echo=False)
        self._underlying_price_df = self._get_refreshed_underlying_price_df()
        self._signal = None
        self._weight = None
        self._portfolio_eligibility_df = None

    def get_back_test(self):
        logger.debug('Fetching data needed for back test ...')
        if self._weight is None:
            raise ValueError('No weight object assigned to index.')
        elif self._signal is None:
            logger.info('No signal object has been assigned to the Index object so a default signal is created.')
            # create a temporary DataFrame filled with 1s where there is a price available.
            rolling_avg_underlying_price_df = self._underlying_price_df\
                .rolling(window=Index._price_availability_observation_threshold, min_periods=1).mean()
            temp_signal_df = pd.DataFrame(data=np.where(rolling_avg_underlying_price_df.isna(), 0, 1),
                                          index=self._underlying_price_df.index,
                                          columns=self._underlying_price_df.columns)
            if self.portfolio_eligibility_df is not None:
                logger.debug('Apply eligibility filter.')
                temp_signal_df *= self.portfolio_eligibility_df
            weights_df = self._get_weight_df(temp_signal_df)  # temporarily set a signal_df and calculate weight_df
            self.weight.signal_df = None  # reset it back
        else:
            weights_df = self._get_weight_df()
        logger.debug('Observe the weights at the rebalancing dates.')
        weights_reb_cal_df = select_rows_from_dataframe_based_on_sub_calendar(weights_df, self.rebalancing_calendar)
        logger.debug('Calculate the back test.')
        index_df = index_calculation(self._underlying_price_df,
                                     weights_reb_cal_df,
                                     self.transaction_cost,
                                     self.fee,
                                     self.initial_amount)
        logger.debug('Back test done.')
        return index_df

    def _get_signal_df(self):
        logger.debug('Retrieve signal DataFrame.')
        return self.signal.get_signal()

    def _get_weight_df(self, custom_signal_df: pd.DataFrame = None):
        logger.debug('Retrieve weight DataFrame.')
        if custom_signal_df is None:
            custom_signal_df = self._get_signal_df()
        self.weight.signal_df = custom_signal_df
        return self.weight.get_weights()

    def _set_data_for_signal(self):
        """Feed required data to the signal object"""
        if is_price_based_signal_subclass_instance(self.signal):
            # assign a price DataFrame to the signal object
            if self.signal.total_return_flag:
                start_date = pd.to_datetime(self._underlying_price_df.index.values[0])
                end_date = pd.to_datetime(self._underlying_price_df.index.values[-1])
                total_return_prices = self._financial_database.get_total_return_df(tickers=self.tickers,
                                                                                   start_date=start_date,
                                                                                   end_date=end_date)
                self.signal.underlying_price_df = total_return_prices
            else:
                self.signal.underlying_price_df = self._underlying_price_df

    def _refresh_underlying_price_df(self):
        # if tickers have changed, 1st date in the reblancing calendar is before the price calendar or end date is after
        #
        tickers_have_changed = self.tickers != list(self._underlying_price_df.columns.values)
        # todo remove - timedelta(days=2 * 365)
        dates_are_not_contained = self.rebalancing_calendar[0] - timedelta(days=2 * 365) \
                                  < self._underlying_price_df.index.values[0]
        end_date_after_available_price = np.datetime64(self.end_date) > self._underlying_price_df.index.values[-1]
        if tickers_have_changed or dates_are_not_contained or end_date_after_available_price:
            self._underlying_price_df = self._get_refreshed_underlying_price_df()
            self.portfolio_eligibility_df = None
            self._set_data_for_signal()
        elif np.datetime64(self.end_date) < self._underlying_price_df.index.values[-1]:
            # todo is this OK?
            end = self._underlying_price_df.index.searchsorted(self.end_date)
            self._underlying_price_df = self._underlying_price_df.copy().loc[:end]
            if self.portfolio_eligibility_df is not None:
                self.portfolio_eligibility_df = self.portfolio_eligibility_df.copy().loc[:end]
            self.end_date = self._underlying_price_df.index.values[-1]
            self._set_data_for_signal()
        else:
            logger.debug('No need to refresh underlying price DataFrame.')

    def _get_refreshed_underlying_price_df(self):
        logger.debug('Refresh underlying price DataFrame.')
        # todo remove rows where all columns are nan
        if self.total_return:
            underlying_price = self._financial_database.get_total_return_df(tickers=self.tickers,
                                                                            start_date=self.rebalancing_calendar[0]
                                                                                       - timedelta(days=2 * 365),
                                                                            end_date=self.end_date,
                                                                            withholding_tax=self.tax_on_dividend)
        else:
            underlying_price = self._financial_database.get_close_price_df(tickers=self.tickers,
                                                                           start_date=self.rebalancing_calendar[0]
                                                                                      - timedelta(days=2 * 365),
                                                                           end_date=self.end_date)
        underlying_price.dropna(how='all', inplace=True)
        return underlying_price

    # ------------------------------------------------------------------------------------------------------------------
    # Decorators: getters

    @property
    def underlying_price_df(self):
        return self._underlying_price_df

    @property
    def signal(self):
        return self._signal

    @property
    def weight(self):
        return self._weight

    @property
    def tickers(self):
        return self._tickers

    @property
    def end_date(self):
        return self._end_date

    @property
    def rebalancing_calendar(self):
        return self._rebalancing_calendar

    @property
    def portfolio_eligibility_df(self):
        return self._portfolio_eligibility_df

    @property
    def tax_on_dividend(self):
        return self._tax_on_dividend

    @property
    def fee(self):
        return self._fee

    @property
    def transaction_cost(self):
        return self._transaction_cost

    @property
    def initial_amount(self):
        return self._initial_amount

    # ------------------------------------------------------------------------------------------------------------------
    # Decorators: setters

    @signal.setter
    def signal(self, signal):
        if is_signal_subclass_instance(signal):
            signal.portfolio_eligibility_df = self.portfolio_eligibility_df
            self._signal = signal
            self._set_data_for_signal()
            logger.debug('Setting signal to: \n{}'.format(signal))
        else:
            raise ValueError('Object is not a sub-class of _Signal.')

    @weight.setter
    def weight(self, weight):
        if is_weight_subclass_instance(weight):
            logger.debug('Setting weight to: \n{}'.format(weight))
            self._weight = weight
        else:
            raise ValueError('Object is not a sub-class of _Weight.')

    @tickers.setter
    def tickers(self, tickers: list):
        tickers.sort()
        logger.debug('Setting tickers (#{}) to: \n{}'.format(len(tickers), tickers))
        self._tickers = tickers
        self._refresh_underlying_price_df()

    @end_date.setter
    def end_date(self, end_date: datetime):
        logger.debug('Setting end date to {}.'.format(str(end_date)[:10]))
        self._end_date = end_date
        if self.rebalancing_calendar[-1] > end_date:
            new_end_date_in_calendar = self.rebalancing_calendar.asof(end_date)
            try:
                location_of_new_end_date = self.rebalancing_calendar.get_loc(new_end_date_in_calendar)
                logger.info('{} rebalancing dates that are before {} (end date) gets removed from the rebalancing '
                            'calendar'.format(len(self.rebalancing_calendar)
                                              - len(self.rebalancing_calendar[:location_of_new_end_date + 1]),
                                              str(end_date)[:10]))
                self._rebalancing_calendar = self.rebalancing_calendar[:location_of_new_end_date + 1]
            except KeyError:
                raise ValueError('end_date is before the first date of the rebalancing calendar.')
        self._refresh_underlying_price_df()

    @rebalancing_calendar.setter
    def rebalancing_calendar(self, rebalancing_calendar: pd.DatetimeIndex):
        if rebalancing_calendar[-1] <= self.end_date:
            logger.debug('Setting rebalancing calendar to: \n{}.'.format(rebalancing_calendar.values))
            self._rebalancing_calendar = rebalancing_calendar
        else:
            raise ValueError('end_date occurs before the end of the rebalance calendar.')
        self._refresh_underlying_price_df()

    @portfolio_eligibility_df.setter
    def portfolio_eligibility_df(self, portfolio_eligibility_df: pd.DataFrame):
        logger.debug('Setting portfolio eligibility DataFrame.')
        self._portfolio_eligibility_df = portfolio_eligibility_df
        if self.signal is not None:
            self.signal.portfolio_eligibility_df = portfolio_eligibility_df

    @tax_on_dividend.setter
    def tax_on_dividend(self, tax_on_dividend: float):
        if tax_on_dividend < 0.0 or tax_on_dividend > 1.0:
            raise ValueError('Dividend tax needs to be a number between 0% and 100%.')
        logger.debug('Setting tax rate on dividends to {}%.'.format(round(tax_on_dividend * 100, 2)))
        self._tax_on_dividend = tax_on_dividend

    @fee.setter
    def fee(self, fee: float):
        if fee < 0.0:
            raise ValueError('Transaction costs needs to be greater or equal to 0%.')
        logger.debug('Setting running index fee p.a. to {}%.'.format(round(fee * 100, 2)))
        self._fee = fee

    @transaction_cost.setter
    def transaction_cost(self, transaction_cost: float):
        if transaction_cost < 0.0:
            raise ValueError('Transaction costs needs to be greater or equal to 0%.')
        logger.debug('Setting transaction costs to {}%.'.format(round(transaction_cost * 100, 2)))
        self._transaction_cost = transaction_cost

    @initial_amount.setter
    def initial_amount(self, initial_amount: float):
        if initial_amount <= 0.0:
            raise ValueError('Initial amount needs to be strictly greater than zero.')
        logger.debug('Setting initial amount to {}.'.format(initial_amount))
        self._initial_amount = initial_amount

    def __repr__(self):
        if self.total_return:
            index_type = 'total '
            if self.tax_on_dividend > 0:
                index_type += ' net '
            else:
                index_type += ' gross '
        else:
            index_type = 'price '
        index_type += 'return'
        description_str = "<Index(type = {}, fee p.a. = {}%, transaction costs = {}%)>".format(index_type,
                                                                                               round(100 * self.fee, 2),
                                                                                               round(100 * self.transaction_cost, 2))
        description_str += '\nSignal: {}\nWeighting method: {}'.format(self.signal, self.weight)
        return description_str


def main():
    # folder_path = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies'
    # calendar
    start_date = '2017-1-15'
    rebalance_calendar = pd.bdate_range(start=start_date,
                                        periods=8,
                                        freq='30D')
    end_date = rebalance_calendar[-1] + timedelta(days=10)
    # tickers
    fin_db = FinancialDatabase(my_database_name, database_echo=False)
    tickers = fin_db.get_ticker({Underlying.currency: 'SEK', Underlying.industry: 'BIOTECHNOLOGY'})
    tickers = tickers[1: 3]

    main_index = Index(rebalance_calendar, tickers, end_date)
    main_index.end_date = datetime(2018, 7, 15)
    sma_signal = SimpleMovingAverageCross(leading_window=20, lagging_window=50)
    equal_weight = EqualWeight()
    # main_index.signal = sma_signal
    main_index.weight = equal_weight
    bt_1 = main_index.get_back_test()

    # liquidity constraints
    underlying_price_df = main_index.underlying_price_df.copy()
    start_date = pd.to_datetime(underlying_price_df.index.values[0])
    end_date = pd.to_datetime(underlying_price_df.index.values[-1])
    liquidity_df = fin_db.get_liquidity_df(list(underlying_price_df),
                                           start_date=start_date,
                                           end_date=end_date)
    avg_liquidity_df = rolling_average(liquidity_df, 60).fillna(0)
    liquidity_threshold = 400000
    eligable_for_investment = pd.DataFrame(np.where(avg_liquidity_df.values > liquidity_threshold, 1, 0),
                                           index=avg_liquidity_df.index,
                                           columns=avg_liquidity_df.columns)
    main_index.portfolio_eligibility_df = eligable_for_investment
    bt_2 = main_index.get_back_test()
    main_index.signal = sma_signal
    bt_3 = main_index.get_back_test()
    main_index.portfolio_eligibility_df = None
    bt_4 = main_index.get_back_test()


if __name__ == '__main__':
    main()




