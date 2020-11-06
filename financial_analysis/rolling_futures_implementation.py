from datetime import datetime

from database.financial_database import BloombergFeeder
from database.config_database import __MY_DATABASE_NAME__
from financial_analysis.rolling_futures_index import __COMMODITY_BBG_UNDERLYING_CODE__, __ROLLING_MONTHS_COMMODITY_CODE__, get_investment_universe

__ORDINAL_NUMBERS__ = {1: 'st', 2: 'nd', 3: 'rd'}


def update_financial_database(tickers: list, bbg_fin_db: BloombergFeeder)->None:

    # add tickers that does not exist in the database
    bbg_fin_db.add_underlying(tickers, refresh_data_after_adding_underlying=False)

    # refresh the tickers
    bbg_fin_db.refresh_data_for_tickers(tickers)


def strategy_ticker_and_description(commodity_symbol: str, allocation_dict: dict):
    # adjust input
    commodity_symbol = commodity_symbol.upper()
    desc = __COMMODITY_BBG_UNDERLYING_CODE__[commodity_symbol] + ' rolling futures strategy. '
    ticker = __COMMODITY_BBG_UNDERLYING_CODE__[commodity_symbol].upper().replace(' ', '_')
    for position in list(allocation_dict.keys()):
        if allocation_dict[position] > 0:
            long_short_str = 'LONG'
        else:
            long_short_str = 'SHORT'
        ticker += '_{}_{}_FUT'.format(long_short_str, position)
        desc += '{} {} {}{} future'.format(long_short_str, str(round(abs(allocation_dict[position])*100, 2)) + '%',
                                           position, __ORDINAL_NUMBERS__.get(position, 'th'))
        if position == list(allocation_dict.keys())[-1]:
            desc += '.'
        else:
            desc += ' ,'
    return desc, ticker + ' INDEX'


def save_index_result_in_database():
    pass


def main():
    # parameters
    commodity_symbols = ['CO']  # get all the tickers for each commodity
    weight_dict = {1: 1.0}
    get_future_tickers_from_bloomberg = True  # not needed to retrieve all the tickers very frequently
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2020, 2, 1)

    bbg_fin_db = BloombergFeeder(__MY_DATABASE_NAME__)
    rolling_future_strategies = None  # the resulting time series for each commodity strategy gets saved here

    for com_symbol in commodity_symbols:
        com_symbol = com_symbol.upper()
        if get_future_tickers_from_bloomberg:
            # add and refresh Bloomberg tickers that does not exist in the database
            all_bbg_future_tickers = bbg_fin_db.bbg_con.get_futures_chain(com_symbol)  # all contracts from Bloomberg
            update_financial_database(all_bbg_future_tickers, bbg_fin_db)
        month_codes = __ROLLING_MONTHS_COMMODITY_CODE__[com_symbol]  # month codes
        relevant_future_tickers = get_investment_universe(com_symbol, month_codes, start_date, end_date)  # tickers for each commodity group
        update_financial_database(relevant_future_tickers, bbg_fin_db)  # update the data in the financial database

        # future_strategy = calculate_rolling_future_index()  # calculate the strategy
        desc, ticker = strategy_ticker_and_description(com_symbol, weight_dict)


if __name__ == '__main__':
    main()
