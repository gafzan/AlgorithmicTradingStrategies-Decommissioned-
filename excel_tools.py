import pandas as pd
from pathlib import Path
import sys
import os


# -----------------------------------------------------------------------------------------------
# excel_tools.py
# -----------------------------------------------------------------------------------------------


def save_df(df_list: {list, pd.DataFrame}, workbook_name: str, folder_path: str, sheet_name_list: {list, str}='Sheet1'):
    """Assumes df_list is a list of DataFrame, name is a string and sheet_name_list is a list of strings
    Saves all DataFrames in df_list to a sheet with the corresponding name in sheet_list"""
    file_name = '\\' + workbook_name + '.xlsx'
    writer = pd.ExcelWriter(folder_path + file_name, engine='xlsxwriter')
    if type(df_list) == type(sheet_name_list) == list:
        if len(df_list) == len(sheet_name_list):
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_name_list[i])  # write the DataFrame into excel
        else:
            raise ValueError('\"df_list\" and \"sheet_name_list\" are not of the same length')
    elif type(df_list) == list or type(sheet_name_list) == list:
        raise ValueError('\"df_list\" and \"sheet_name_list\" are not of the same type')
    else:
        df_list.to_excel(writer, sheet_name=sheet_name_list)  # write the DataFrame into excel
    writer.save()  # close the Pandas Excel writer and output the excel file


def load_df(workbook_name: str, folder_path: str, sheet_name: str = 'Sheet1', first_column_index=True):
    """Assumes path_name and file_name are strings
    Returns a DataFrame"""
    base_path = Path(folder_path)
    excel_file_name = workbook_name + '.xlsx'
    xl_file_fullname = base_path / excel_file_name
    if not xl_file_fullname.exists():
        print('No such file or directory!')
        sys.exit(-1)
    if first_column_index:
        data = pd.read_excel(xl_file_fullname, index_col=0, sheet_name=sheet_name)
    else:
        data = pd.read_excel(xl_file_fullname, sheet_name=sheet_name)
    return data


def excel_files_in_directory(directory: str) -> list:
    """Assumes that directory is a string showing the directory path. Returns a list of strings of the name of all files
    ending with '.xlsx'"""
    excel_file_names = []
    files_in_dir = os.listdir(directory)
    for file in files_in_dir:
        if file.endswith('.xlsx'):
            excel_file_names.append(file.replace('.xlsx', ''))
    return excel_file_names


def choose_excel_file_from_folder(folder_path: str) -> str:
    """Assumes folder_path is a string. Returns a string of the excel file chosen by the user from the folder"""
    # print the available excel files
    counter = 0
    excel_name_list = excel_files_in_directory(folder_path)
    for excel_name in excel_name_list:
        counter += 1
        print('{}: {}'.format(counter, excel_name))

    # chose the excel file
    ask_user = True
    number = 1
    while ask_user:
        number = int(input('Enter a number between 1 and {}:'.format(counter)))
        ask_user = not isinstance(number, int) or 1 > number or number > counter
    return excel_name_list[number - 1]


def format_risk_return_analysis_workbook(complete_workbook_path: str):
    pass

