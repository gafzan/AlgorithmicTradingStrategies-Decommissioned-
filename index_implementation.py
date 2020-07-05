import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import datetime, date

import logging

# my modules
from index import Index
from index_weight import EqualWeight
from index_signal import VolatilityRankSignal, SimpleMovingAverageCrossSignal
from finance_tools import return_and_risk_analysis, plot_results, rolling_average
from financial_database import FinancialDatabase, Underlying
from config_database import my_database_name, back_test_folder
from excel_tools import save_df, format_risk_return_analysis_workbook

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Parameters
    start_date = datetime(2007, 1, 1)
    end_date = date.today()  # datetime(2019, 12, 30)
    rebalance_frequency = '20D'
    index_fee = 0.0  # % p.a.
    index_transaction_cost = 0.00  # %
    liquidity_lag = 60
    liquidity_threshold = 1e6
    vol_lag = 60
    benchmark = '^OMX'
    # leading_window = 1
    # lagging_window = 50

    fin_db_handler = FinancialDatabase(my_database_name, database_echo=False)
    save = True
    save_underlying_data = False  # for sanity check
    index_name = 'LOW_VOL_EQW_INDEX'  # 'Trend following global market 25% short SEK'
    file_path = back_test_folder
    plot = True

    # ------------------------------------------------------------------------------------------------------------------
    # Rebalancing calendar
    rebalance_calendar = pd.bdate_range(start=start_date, end=end_date, freq=rebalance_frequency)

    # ------------------------------------------------------------------------------------------------------------------
    # Get ticker
    ticker_list = fin_db_handler.get_ticker({Underlying.underlying_type: 'EQUITY', Underlying.exchange: 'STO',
                                             Underlying.currency: 'SEK'})

    # ------------------------------------------------------------------------------------------------------------------
    # Index setup
    main_index = Index(tickers=ticker_list, rebalancing_calendar=rebalance_calendar, index_fee=index_fee / 100,
                       transaction_cost=index_transaction_cost / 100)

    # ------------------------------------------------------------------------------------------------------------------
    # Index eligibility
    liquidity_df = fin_db_handler.get_liquidity_df(ticker_list, start_date=start_date - BDay(liquidity_lag + 10),
                                                   end_date=end_date, currency='SEK')
    avg_liquidity_df = rolling_average(liquidity_df, liquidity_lag)
    avg_liquidity_df = avg_liquidity_df[start_date:]
    eligibility_df = (avg_liquidity_df > liquidity_threshold) * 1.0

    # ------------------------------------------------------------------------------------------------------------------
    # Index weights
    equal_weight = EqualWeight()
    logger.debug('Assigns weight object to index.')
    main_index.weight = equal_weight

    # ------------------------------------------------------------------------------------------------------------------
    # Index signal
    # sma_signal = SimpleMovingAverageCrossSignal((leading_window, lagging_window), ticker_list=ticker_list)
    # logger.debug('Assigns signal object to index.')
    # main_index.signal = sma_signal
    low_vol_signal = VolatilityRankSignal(vol_lag, eligibility_df=eligibility_df, total_return=False, rank_number=25)
    main_index.signal = low_vol_signal

    # ------------------------------------------------------------------------------------------------------------------
    # Index back test
    logger.debug('Perform back test.')
    back_test_df = main_index.get_back_test(end_date=end_date, return_index_only=False)
    back_test_df.rename(columns={'index': index_name}, inplace=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Benchmark
    benchmark_price_df = fin_db_handler.get_close_price_df(benchmark, start_date, end_date)
    if benchmark is not None:
        has_benchmark = True
    else:
        has_benchmark = False
    back_test_df = back_test_df.join(benchmark_price_df)

    # ------------------------------------------------------------------------------------------------------------------
    # Weight analysis
    back_test_at_rebalance_df = back_test_df[back_test_df['rebalance'] == 1].copy()  # select rebalance date rows
    weight_df = back_test_at_rebalance_df.loc[:, back_test_at_rebalance_df.columns.str.endswith('_weight')]
    weight_df.columns = [col_name.replace('_weight', '') for col_name in list(weight_df)]  # tickers as column names

    weight_unstacked_df = weight_df.unstack().reset_index()
    weight_unstacked_df.columns = ['ticker', 'date', 'weight']

    # country
    ticker_country_dict = fin_db_handler.get_ticker_underlying_attribute_dict(list(weight_df), Underlying.country)
    weight_unstacked_df['country'] = weight_unstacked_df['ticker'].map(ticker_country_dict)
    country_weight_df = weight_unstacked_df.drop(['ticker'], axis=1)
    country_weight_df = pd.pivot_table(country_weight_df, values='weight', index='date', columns='country', aggfunc=np.sum)

    # sector
    ticker_sector_dict = fin_db_handler.get_ticker_underlying_attribute_dict(list(weight_df), Underlying.sector)
    weight_unstacked_df['sector'] = weight_unstacked_df['ticker'].map(ticker_sector_dict)
    sector_weight_df = weight_unstacked_df.drop(['ticker'], axis=1)
    sector_weight_df = pd.pivot_table(sector_weight_df, values='weight', index='date', columns='sector', aggfunc=np.sum)

    # industry
    ticker_industry_dict = fin_db_handler.get_ticker_underlying_attribute_dict(list(weight_df), Underlying.industry)
    weight_unstacked_df['industry'] = weight_unstacked_df['ticker'].map(ticker_industry_dict)
    industry_weight_df = weight_unstacked_df.drop(['ticker'], axis=1)
    industry_weight_df = pd.pivot_table(industry_weight_df, values='weight', index='date', columns='industry', aggfunc=np.sum)

    index_result_df_dict = return_and_risk_analysis(back_test_df[[index_name, benchmark]].copy(), print_results=True,
                                                    has_benchmark=has_benchmark)

    # sanity check
    if save_underlying_data:
        logger.debug('Saving underlying data.')
        underlying_df = main_index.basket_prices()
        rebalance_calendar_df = pd.DataFrame(list(rebalance_calendar))
        signal = main_index.signal.get_signal_df()
        weight = main_index.weight.get_weights()
        folder_path = file_path + '\\' + index_name + '_underlying_data' + f' - {date.today()}' + '.xlsx'
        save_df([underlying_df, rebalance_calendar_df, signal, weight, back_test_df], full_path=folder_path,
                sheet_name_list=['price', 'rebalance calendar', 'signal', 'weight', 'back test'])

    # ------------------------------------------------------------------------------------------------------------------
    # Plot and save index result in Excel
    if save:
        logger.debug('Saving index back test.')
        index_result_df_dict.update({'Weight': weight_df})
        index_result_df_dict.update({'Sector weight': sector_weight_df})
        index_result_df_dict.update({'Industry weight': industry_weight_df})
        index_result_df_dict.update({'Country weight': country_weight_df})
        folder_path = file_path + '\\' + index_name + f' - {date.today()}' + '.xlsx'
        save_df(list(index_result_df_dict.values()), full_path=folder_path, sheet_name_list=list(index_result_df_dict.keys()))

        # format the workbook
        logger.debug('Formatting saved excel file.')
        format_risk_return_analysis_workbook(folder_path)

    if plot:
        plot_results(index_result_df_dict)


if __name__ == '__main__':
    main()
