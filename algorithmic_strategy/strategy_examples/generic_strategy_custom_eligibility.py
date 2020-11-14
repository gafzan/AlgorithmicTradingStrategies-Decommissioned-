"""
generic_strategy_custom_eligibility.py

This can be seen as a algorithmic strategy back test implementation template.

1) Define your parameters

2) Load an 'investment universe' from Excel and apply other filters if necessary

3) Define the components of your index such as signals (e.g. SMA crossovers or low volatility), weights (e.g. Minimum
Variance) and overlays (e.g. beta hedge that shorts a specific market instrument)

4) Combine the index components from 3) and apply it to the investment universe from 2) and perform the back test

5) Analyse the result through risk & return metrics (e.g. maximum drawdown), through plots and save the results in excel
for future reference

"""
import pandas as pd

# my modules
# strategy creation
from algorithmic_strategy.investment_universe import InvestmentUniverse
import algorithmic_strategy.strategy_signal as algo_signal
import algorithmic_strategy.strategy_weight as algo_weight
import algorithmic_strategy.strategy_overlay as algo_overlay
from algorithmic_strategy.strategy import Index

# strategy analysis
from excel_tools import save_df, format_risk_return_analysis_workbook, load_df_from_chosen_excel_file, load_df
from financial_analysis.finance_tools import return_and_risk_analysis, plot_results
from database.config_database import __BACK_TEST_FOLDER__, __TICKER_ELIGIBILITY_FOLDER__


def get_eligibility_df(full_path: str = None):
    if full_path is None:
        eligibility_df = load_df_from_chosen_excel_file(initialdir=__TICKER_ELIGIBILITY_FOLDER__,
                                                        title_msg='Select an excel file containing the eligibility data',
                                                        sheet_name='eligibility')
    else:
        eligibility_df = load_df(full_path=full_path)
    return eligibility_df


def main():
    # 1) parameters of the strategy
    observation_calendar = pd.date_range(start='1 jan 2008',
                                         end='2020',
                                         freq='m')
    volatility_target = 0.10  # used by a volatility target overlay
    volatility_target_obs_lag = 60  # used by a volatility target overlay
    index_name = 'My Index'
    full_path_to_eligibility_dataframe = None
    initial_fiter_desc = 'custom filter'
    save_back_test_to_excel = True
    print_performance_result = True
    make_plot = True

    # 2) setup the investment universe
    initial_eligibility_df = get_eligibility_df(full_path=full_path_to_eligibility_dataframe)
    tickers = list(initial_eligibility_df)
    investment_universe = InvestmentUniverse(
        tickers=tickers,
        observation_calendar=observation_calendar
    )
    investment_universe.apply_custom_filter(custom_eligibility_df=initial_eligibility_df, filter_desc=initial_fiter_desc)
    # (some examples of other filters)
    # investment_universe.apply_published_close_price_filter(max_number_days_since_publishing=5)
    # investment_universe.apply_close_price_history_filter(100)

    # 3) define the index components (signals, weight and overlays)
    signal_component = algo_signal.PerformanceRankSignal(
        performance_observation_period=[40, 60],
        rank_number=1
    )
    weight_component = algo_weight.EqualWeight()
    overlay_component = algo_overlay.VolatilityControl(
        vol_control_level=volatility_target,
        vol_lag=volatility_target_obs_lag
    )

    # 4) setup the index and perform back test
    index = Index(
        investment_universe=investment_universe,
        signal=signal_component,
        weight=weight_component,
        overlay=overlay_component,
    )
    back_test_df = index.get_back_test()
    back_test_df.columns = [index_name]

    # 5) analyse the results
    performance_data_dict = {'Description': index.get_index_desc_df()}
    performance_data_dict.update(
        return_and_risk_analysis(
            underlying_price_df=back_test_df,
            print_results=print_performance_result
        )
    )

    if save_back_test_to_excel:
        # save all the DataFrames into excel and reformat the result
        file_path = __BACK_TEST_FOLDER__ + '\\' + index_name + '.xlsx'
        save_df(
            df_list=list(performance_data_dict.values()),
            full_path=file_path,
            sheet_name_list=list(performance_data_dict.keys())
        )
        format_risk_return_analysis_workbook(complete_workbook_path=file_path)

    if make_plot:
        plot_results(performance_data_dict)


if __name__ == '__main__':
    main()


