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
# contains excel workbooks with tickers
excel_ticker_folder = r'C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies\excel_data\tickers'.format(user_name)
create_directory(excel_ticker_folder)
# contains excel workbooks with data requested from the database
data_request_folder = r'C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies\excel_data\data_requests'.format(user_name)
create_directory(data_request_folder)
# contains excel workbooks used to insert data into the database
excel_files_to_feed_database_folder = r'C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies\excel_data\excel_based_feeder_workbooks'.format(user_name)
create_directory(excel_files_to_feed_database_folder)

my_database_name = r'sqlite:///C:\Users\{}\PycharmProjects\AlgorithmicTradingStrategies\financial_database_v1.db'.format(user_name)

