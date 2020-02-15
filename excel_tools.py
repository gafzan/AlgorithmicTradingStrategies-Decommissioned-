import pandas as pd
from pathlib import Path
import sys
import os
from openpyxl import load_workbook
from openpyxl.styles.borders import Border, Side
from openpyxl.styles import Alignment, Font, PatternFill, Color
from openpyxl.utils import get_column_letter


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
    """Assumes that complete_workbook_path is a string containing the path to the Excel workbook. Changes the format
    of certain sheets."""
    workbook = load_workbook(complete_workbook_path)
    border = Border(left=Side(border_style='thin', color='000000'), right=Side(border_style='thin', color='000000'),
                    top=Side(border_style='thin', color='000000'), bottom=Side(border_style='thin', color='000000'))

    COLUMN_NAMES_FONT = Font(bold=True,
                             italic=False,
                             vertAlign=None,
                             underline='none',
                             strike=False,
                             color='FFFFFF')

    fill_color = Color(rgb='00B050')
    COLUMN_NAMES_FILL = PatternFill(patternType='solid', fgColor=fill_color)

    # check if the necessary sheets exists (some sheets are optional)
    try:
        # format the performance sheet
        performance_sheet = workbook['Performance']
        performance_sheet.column_dimensions['A'].width = 12
        performance_sheet.cell(row=1, column=1).value = 'Date'
        performance_sheet.cell(row=1, column=1).font = COLUMN_NAMES_FONT
        performance_sheet.cell(row=1, column=1).fill = COLUMN_NAMES_FILL
        for row in range(1, performance_sheet.max_row + 1):
            performance_sheet.cell(row=row, column=1).number_format = 'D MMM YYYY'
            performance_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='center', vertical='center')
            performance_sheet.cell(row=row, column=1).border = border
            for col in range(2, performance_sheet.max_column + 1):
                performance_sheet.cell(row=row, column=col).number_format = '#,##0.00'
                performance_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center',
                                                                                  vertical='center')
                performance_sheet.cell(row=1, column=col).font = COLUMN_NAMES_FONT
                performance_sheet.cell(row=1, column=col).fill = COLUMN_NAMES_FILL
                performance_sheet.cell(row=row, column=col).border = border
        performance_sheet.freeze_panes = 'B2'
    except KeyError:
        raise ValueError("There is no sheet named 'Performance'")
    try:
        # format the risk and return sheet
        risk_return_sheet = workbook['Risk and return']
        risk_return_sheet.column_dimensions['A'].width = 19
        for row in range(1, risk_return_sheet.max_row + 1):
            risk_return_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='left', vertical='center')
            risk_return_sheet.cell(row=row, column=1).border = border
            risk_return_sheet.cell(row=1, column=1).font = COLUMN_NAMES_FONT
            risk_return_sheet.cell(row=1, column=1).fill = COLUMN_NAMES_FILL
            for col in range(2, performance_sheet.max_column + 1):
                if row == 4:  # Sharpe ratio is not shown as %
                    risk_return_sheet.cell(row=row, column=col).number_format = '#,##0.00'
                else:
                    risk_return_sheet.cell(row=row, column=col).number_format = '0.00%'
                risk_return_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center',
                                                                                  vertical='center')
                risk_return_sheet.cell(row=1, column=col).font = COLUMN_NAMES_FONT
                risk_return_sheet.cell(row=1, column=col).fill = COLUMN_NAMES_FILL
                risk_return_sheet.cell(row=row, column=col).border = border
    except KeyError:
        raise ValueError("There is no sheet named 'Risk and return'")
    try:
        # format rolling 1Y return sheet
        rolling_return_sheet = workbook['Rolling 1Y return']
        rolling_return_sheet.column_dimensions['A'].width = 12
        rolling_return_sheet.cell(row=1, column=1).value = 'Date'
        for row in range(1, rolling_return_sheet.max_row + 1):
            rolling_return_sheet.cell(row=row, column=1).number_format = 'D MMM YYYY'
            rolling_return_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='center', vertical='center')
            rolling_return_sheet.cell(row=row, column=1).border = border
            rolling_return_sheet.cell(row=1, column=1).font = COLUMN_NAMES_FONT
            rolling_return_sheet.cell(row=1, column=1).fill = COLUMN_NAMES_FILL
            for col in range(2, rolling_return_sheet.max_column + 1):
                rolling_return_sheet.cell(row=row, column=col).number_format = '0.00%'
                rolling_return_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center',
                                                                                     vertical='center')
                rolling_return_sheet.cell(row=1, column=col).font = COLUMN_NAMES_FONT
                rolling_return_sheet.cell(row=1, column=col).fill = COLUMN_NAMES_FILL
                rolling_return_sheet.cell(row=row, column=col).border = border
        rolling_return_sheet.freeze_panes = 'B2'
    except KeyError:
        raise ValueError("There is no sheet named 'Rolling 1Y return'")
    try:
        # format rolling 1Y volatility sheet
        rolling_volatility_sheet = workbook['Rolling 1Y volatility']
        rolling_volatility_sheet.column_dimensions['A'].width = 12
        rolling_volatility_sheet.cell(row=1, column=1).value = 'Date'
        for row in range(1, rolling_volatility_sheet.max_row + 1):
            rolling_volatility_sheet.cell(row=row, column=1).number_format = 'D MMM YYYY'
            rolling_volatility_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='center',
                                                                                   vertical='center')
            rolling_volatility_sheet.cell(row=row, column=1).border = border
            rolling_volatility_sheet.cell(row=1, column=1).font = COLUMN_NAMES_FONT
            rolling_volatility_sheet.cell(row=1, column=1).fill = COLUMN_NAMES_FILL
            for col in range(2, rolling_volatility_sheet.max_column + 1):
                rolling_volatility_sheet.cell(row=row, column=col).number_format = '0.00%'
                rolling_volatility_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center',
                                                                                         vertical='center')
                rolling_volatility_sheet.cell(row=1, column=col).font = COLUMN_NAMES_FONT
                rolling_volatility_sheet.cell(row=1, column=col).fill = COLUMN_NAMES_FILL
                rolling_volatility_sheet.cell(row=row, column=col).border = border
        rolling_volatility_sheet.freeze_panes = 'B2'
    except KeyError:
        raise ValueError("There is no sheet named 'Rolling 1Y volatility'")
    try:
        # format rolling 1Y drawdown sheet
        rolling_drawdown_sheet = workbook['Rolling 1Y drawdown']
        rolling_drawdown_sheet.column_dimensions['A'].width = 12
        rolling_drawdown_sheet.cell(row=1, column=1).value = 'Date'
        for row in range(1, rolling_drawdown_sheet.max_row + 1):
            rolling_drawdown_sheet.cell(row=row, column=1).number_format = 'D MMM YYYY'
            rolling_drawdown_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='center', vertical='center')
            rolling_drawdown_sheet.cell(row=row, column=1).border = border
            rolling_drawdown_sheet.cell(row=1, column=1).font = COLUMN_NAMES_FONT
            rolling_drawdown_sheet.cell(row=1, column=1).fill = COLUMN_NAMES_FILL
            for col in range(2, rolling_drawdown_sheet.max_column + 1):
                rolling_drawdown_sheet.cell(row=row, column=col).number_format = '0.00%'
                rolling_drawdown_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center',
                                                                                       vertical='center')
                rolling_drawdown_sheet.cell(row=1, column=col).font = COLUMN_NAMES_FONT
                rolling_drawdown_sheet.cell(row=1, column=col).fill = COLUMN_NAMES_FILL
                rolling_drawdown_sheet.cell(row=row, column=col).border = border
        rolling_drawdown_sheet.freeze_panes = 'B2'
    except KeyError:
        raise ValueError("There is no sheet named 'Rolling 1Y drawdown'")

    # format all monthly return tables by looping through each sheet who's name contains the substring 'Return table'
    return_table_sheet_name_list = [sheet_name for sheet_name in workbook.sheetnames if 'Return table' in sheet_name]
    for return_table_sheet_name in return_table_sheet_name_list:
        return_table_sheet = workbook[return_table_sheet_name]
        return_table_sheet.column_dimensions[get_column_letter(return_table_sheet.max_column)].width = 12
        for row in range(1, return_table_sheet.max_row + 1):
            return_table_sheet.cell(row=row, column=1).alignment = Alignment(horizontal='center', vertical='center')
            return_table_sheet.cell(row=row, column=1).border = border
            for col in range(2, return_table_sheet.max_column + 1):
                return_table_sheet.cell(row=row, column=col).alignment = Alignment(horizontal='center', vertical='center')
                return_table_sheet.cell(row=row, column=col).number_format = '0.00%'
                return_table_sheet.cell(row=row, column=col).border = border

    workbook.save(complete_workbook_path)


def main():
    path = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\excel_data\performance data test - 2020-02-15.xlsx'
    format_risk_return_analysis_workbook(path)


if __name__ == '__main__':
    main()
