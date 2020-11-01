"""
strategy_implementation.py
"""
from matplotlib import pyplot as plt
import pandas as pd

# my modules
from algorithmic_strategy.strategy_OLD import Index
from algorithmic_strategy.strategy_weight import EqualWeight, VolatilityWeight
from algorithmic_strategy.strategy_signal import SimpleMovingAverageCrossSignal
from algorithmic_strategy.investment_universe import InvestmentUniverse

from financial_analysis.finance_tools import return_and_risk_analysis
from excel_tools import save_df, format_risk_return_analysis_workbook
from database.config_database import back_test_folder, my_database_name
from database.financial_database import FinancialDatabase
from dataframe_tools import merge_two_dataframes_as_of


def main():
    # parameters
    benchmark_ticker = '^OMX'
    print_result = True
    plot_result = True
    save_results = True

    # define the investment universe
    tickers = ["BOL.ST", "SCA-B.ST", "AZN.ST", "ELUX-B.ST", "TELIA.ST", "ABB.ST", "ESSITY-B.ST", "ASSA-B.ST", "SSAB-A.ST",
               "HEXA-B.ST", "INVE-B.ST", "HM-B.ST", "VOLV-B.ST", "SAND.ST", "NDA-SE.ST", "SKA-B.ST", "ERIC-B.ST", "KINV-B.ST",
               "SECU-B.ST", "SHB-A.ST", "TEL2-B.ST", "ALFA.ST", "ATCO-A.ST", "ALIV-SDB.ST", "SWED-A.ST", "SWMA.ST", "ATCO-B.ST",
               "SEB-A.ST", "SKF-B.ST", "GETI-B.ST"]
    # TODO why start so late?
    liquid_stock_universe = InvestmentUniverse(tickers=tickers, start='2013', end='2020', freq='3M')
    liquid_stock_universe.apply_liquidity_filter(60, 1)

    # setup the index
    sma_eqw_index = Index(investment_universe=liquid_stock_universe, observation_calendar=pd.date_range('2013', '2020',
                                                                                                        freq='M'))
    sma_eqw_index.weight = EqualWeight(net_zero_exposure=False)
    sma_eqw_index.signal = SimpleMovingAverageCrossSignal(10, 50)
    index_reporting(sma_eqw_index, index_name='SMA', print_results=print_result, save_results=save_results, plot_results=plot_result,
                    benchmark_ticker=benchmark_ticker)


def index_reporting(index_: Index, index_name: str = 'TEST', benchmark_ticker: str = None, save_results: bool = True,
                    print_results: bool = True, plot_results: bool = False, do_formatting: bool = True,
                    save_entire_back_test: bool = False):
    """
    Calculates the back test of an instance of Index object and prints, saves and plots results when applicable.
    :param index_: Index
    :param index_name: str
    :param benchmark_ticker: str
    :param save_results: bool
    :param print_results: bool
    :param plot_results: bool
    :param do_formatting: bool
    :param save_entire_back_test: bool
    :return: None
    """
    # perform back test and add nem as a column head
    back_test_result = index_.get_back_test()
    try:
        index_back_test = back_test_result.loc[:, ['NET_INDEX']].dropna()
    except KeyError:
        index_back_test = back_test_result.loc[:, ['GROSS_INDEX']].dropna()
    index_name = index_name.upper() + '_INDEX'
    index_back_test.columns = [index_name]

    if benchmark_ticker:
        # download price for the benchmark
        fin_db = FinancialDatabase(my_database_name)
        benchmark_price = fin_db.get_close_price_df(benchmark_ticker)
        index_back_test = merge_two_dataframes_as_of(index_back_test, benchmark_price)
        # normalize the benchmark price to start at the same value as the index
        index_back_test.iloc[:, -1] = index_back_test.iloc[0, -2] * index_back_test.iloc[:, -1] / index_back_test.iloc[0, -1]

    # store each DataFrame with results as a dictionary
    performance_data = {'Description': index_.get_index_desc_df()}
    performance_data.update(return_and_risk_analysis(index_back_test, benchmark_ticker is not None, print_results))
    if save_entire_back_test:
        performance_data.update({'Back test': back_test_result})
    if save_results:
        save_df(list(performance_data.values()), workbook_name=index_name, folder_path=back_test_folder,
                sheet_name_list=list(performance_data.keys()))
        if do_formatting:
            format_risk_return_analysis_workbook(back_test_folder + '\\' + index_name + '.xlsx')
    if plot_results:
        index_back_test.plot()
        plt.show()


if __name__ == '__main__':
    main()


