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
base_folder = r'C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies'.format(user_name)
create_directory(base_folder)
# contains excel workbooks with tickers
excel_ticker_folder = base_folder + r'\excel_data' + '\\tickers'
create_directory(excel_ticker_folder)
# contains excel workbooks with data requested from the database
data_request_folder = base_folder + r'\excel_data' + '\\data_requests'
create_directory(data_request_folder)
# contains excel workbooks used to insert data into the database
excel_files_to_feed_database_folder = base_folder + r'\excel_data' + '\\excel_based_feeder_workbooks'
create_directory(excel_files_to_feed_database_folder)

my_database_name = r'sqlite:///' + base_folder + '\\financial_database_v1.db'
