import os
user_name = os.environ.get('USERNAME')


def create_directory(directory: str)-> None:
    """
    Checks if the directory exists. If it does not exists, the directory is created
    :param directory: string
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# folders for accessing and saving data
__BASE_FOLDER__ = r'C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies'.format(user_name)
create_directory(__BASE_FOLDER__)
# contains excel workbooks with tickers
__EXCEL_TICKER_FOLDER__ = __BASE_FOLDER__ + r'\excel_data' + '\\tickers'
create_directory(__EXCEL_TICKER_FOLDER__)
__TICKER_ELIGIBILITY_FOLDER__ = __EXCEL_TICKER_FOLDER__ + '\\ticker_eligibility'
create_directory(__TICKER_ELIGIBILITY_FOLDER__)
# contains excel workbooks with data requested from the database
__DATA_REQUEST_FOLDER__ = __BASE_FOLDER__ + r'\excel_data' + '\\data_requests'
create_directory(__DATA_REQUEST_FOLDER__)
# contains excel workbooks used to insert data into the database
__DATABASE_FEED_EXCEL_FILES_FOLDER__ = __BASE_FOLDER__ + r'\excel_data' + '\\excel_based_feeder_workbooks'
create_directory(__DATABASE_FEED_EXCEL_FILES_FOLDER__)
# contains excel workbooks with back test data
__BACK_TEST_FOLDER__ = __BASE_FOLDER__ + r'\excel_data' + '\\back_tests'
create_directory(__BACK_TEST_FOLDER__)
__MY_DATABASE_NAME__ = r'sqlite:///' + __BASE_FOLDER__ + '\\database\database_files\\tutorial.db'
create_directory(__BASE_FOLDER__ + r'\\database\database_files')
