"""
dataframe_tools.py
"""
import pandas as pd
import numpy as np
from itertools import groupby


def nan_before_dates(df: pd.DataFrame, col_name_date_dict: dict):
    """
    Using a {column name: date} dictionary, set all values before the date given by each column to nan
    :param df: DataFrame
    :param col_name_date_dict: dict {column name: date}
    :return: DataFrame
    """

    # check that all keys in the {column name: date} dictionary are in the column names of df
    if any(key not in list(df) for key in col_name_date_dict.keys()):
        raise ValueError("all keys in 'col_name_date_dict' needs to be in the column names of 'df'")

    rows = np.searchsorted(df.index, list(col_name_date_dict.values()), side='right')
    cols = [list(df).index(key) for key in col_name_date_dict.keys()]

    # set all rows before a certain row to a 'flag number' that will later be replaced by nan
    values = df.copy().to_numpy()
    flag_value = -999
    for row, col in zip(rows, cols):
        values[:row, col] = flag_value

    values = np.where(values == flag_value, np.nan, values)
    return pd.DataFrame(data=values, index=df.index, columns=df.columns)


def dataframe_with_new_calendar(df: pd.DataFrame, new_calendar: pd.DatetimeIndex):
    """
    Returns a new DataFrame where the row data are based on the new calendar (similar to Excel's VLOOKUP with
    approximate match)
    :param df: DataFrame
    :param new_calendar: DatetimeIndex
    :return: DataFrame
    """
    # find the position in the old calendar that closest represents the new calendar dates
    original_calendar = df.index
    date_index_list = np.searchsorted(original_calendar, new_calendar, side='right')
    date_index_list = [d_i - 1 for d_i in date_index_list if d_i > 0]
    data_for_new_calendar = df.to_numpy()[date_index_list, :]

    # in case the first dates in the new calendar are before the first available date in the DataFrame, add nans to the
    # first rows
    if data_for_new_calendar.shape[0] != len(new_calendar):
        num_missing_rows = len(new_calendar) - data_for_new_calendar.shape[0]
        nan_array = np.empty((num_missing_rows, data_for_new_calendar.shape[1]))
        nan_array[:] = np.nan
        # add the data after the nan rows
        data_for_new_calendar = np.vstack([nan_array, data_for_new_calendar])
    return pd.DataFrame(data=data_for_new_calendar, index=new_calendar, columns=df.columns)


def winsorize_dataframe(df: pd.DataFrame, lower_q: float = 0, upper_q: float = 1):
    """
    Returns a new DataFrame where each data is set to nan if it is outside the percentiles. The percentiles are
    calculated per row
    :param df: DataFrame
    :param lower_q: float
    :param upper_q: float
    :return: DataFrame
    """
    if lower_q >= upper_q:
        raise ValueError("'upper_q' ({}) needs to be strictly larger than 'lower_q' ({})".format(upper_q, lower_q))
    df_q = df.quantile(q=[lower_q, upper_q], axis=1).T
    df_q.columns = ['low', 'high']
    # if data is outside the percentiles, set to nan else 1
    win_array = np.where(df.lt(df_q['low'], axis='index') | df.gt(df_q['high'], axis='index'), np.nan, 1)
    return df * win_array


def dataframe_aggregate(df_list: list, agg_mthd: str):
    """
    Returns a new DataFrame that has aggregated the DataFrames in the list element-wise
    :param df_list: DataFrame
    :param agg_mthd: str e.g. 'max' or 'avg'
    :return: DataFrame
    """
    agg_mthd = agg_mthd.lower()
    same_column_and_index(df_list=df_list)  # raises an error if either the columns or the index are not the same
    array_list = [df.to_numpy() for df in df_list]
    agg_array = None
    for array in array_list:
        if agg_array is None:
            agg_array = array
        else:
            if agg_mthd == 'max':
                agg_array = np.maximum(agg_array, array)
            elif agg_mthd == 'min':
                agg_array = np.minimum(agg_array, array)
            elif agg_mthd == 'avg' or 'mean':
                agg_array = np.add(agg_array, array)
            else:
                raise ValueError("agg_mthd='{}' is not a recognized aggregation method".format(agg_mthd))

    # divide the sum with the number of DataFrames if the method is an average
    if agg_mthd == 'avg' or 'mean':
        agg_array /= len(array_list)

    return pd.DataFrame(data=agg_array, columns=list(df_list[0]), index=df_list[0].index)


def same_column_and_index(df_list: list):
    """
    Raises an error if either the columns or the index are not the same
    :param df_list: list
    :return: None
    """
    if any([set(df_list[0].columns) != set(df.columns) for df in df_list]):
        raise ValueError('the columns of the DataFrames in the list are not the same')
    elif any(not df_list[0].index.equals(df.index) for df in df_list):
        raise ValueError('the index of the DataFrames in the list are not the same')
    else:
        return


def equal_in_iterable(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def select_rows_from_dataframe_based_on_sub_calendar(original_data_df: pd.DataFrame,
                                                     date_time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Assumes that original_data_df is a DataFrame with dates as index and date_time_index is a DatetimeIndex.
    Select the rows from original_data_df whose index is closest (but no look ahead bias) to the date_time_index."""
    original_data_reset_df = original_data_df.reset_index()
    col_name_to_merge_on = list(original_data_reset_df)[0]
    sub_date_df = pd.DataFrame(data=date_time_index, columns=[col_name_to_merge_on])
    original_data_on_sub_date_df = pd.merge_asof(sub_date_df, original_data_reset_df, on=col_name_to_merge_on)
    original_data_on_sub_date_df.set_index(date_time_index, inplace=True)
    original_data_on_sub_date_df.drop([col_name_to_merge_on], axis=1, inplace=True)
    return original_data_on_sub_date_df


def dataframe_has_same_index_and_column_names(df_1: {pd.DataFrame, None}, df_2: {pd.DataFrame, None}) -> bool:
    """Assumes that df_1 nd df_2 are both DataFrames. Returns True if the two DataFrames have same index and column
    names and in same order, else returns False."""
    same_col_names = list(df_1) == list(df_2)
    same_index = list(df_1.index.values) == list(df_2.index.values)
    return same_col_names and same_index


def get_unique_values_from_dataframe(values_df: pd.DataFrame) -> list:
    """Assumes that values_df is a DataFrame. Returns a sorted list of the unique values in the DataFrame."""
    unique_value_list = []
    for col_name in list(values_df):
        unique_values_in_col = values_df[col_name].unique()
        new_unique_values = list(set(unique_values_in_col).difference(set(unique_value_list)))
        unique_value_list.extend(new_unique_values)
    return unique_value_list


def dataframe_values_check(values_df: pd.DataFrame, *args) -> bool:
    """Assumes that values_df is a DataFrame. Returns True if all the unique values in values_df exist in *args, else
    return False."""
    unique_values_in_df = set(get_unique_values_from_dataframe(values_df))
    allowed_domain = set(args)
    return len(unique_values_in_df.difference(allowed_domain)) == 0


def merge_two_dataframes_as_of(left_df: pd.DataFrame, right_df: pd.DataFrame, left_suffix: str = '_x'):
    result = left_df.copy()
    right_df = right_df.copy()
    result.reset_index(inplace=True)
    right_df.reset_index(inplace=True)
    index_left_on_col_name = list(result)[0]
    index_right_on_col_name = list(right_df)[0]
    result = pd.merge_asof(result, right_df, left_on=index_left_on_col_name, right_on=index_right_on_col_name,
                           suffixes=('', left_suffix))
    result.set_index(list(result)[0], inplace=True)
    try:
        result.drop([index_left_on_col_name], inplace=True, axis=1)
    except KeyError:
        pass
    try:
        result.drop([index_right_on_col_name], inplace=True, axis=1)
    except KeyError:
        pass
    return result


def rank_filter(df: pd.DataFrame, ascending: bool, inclusive: bool, rank_threshold: {int, float}):
    """
    Returns a DataFrame with 1 if the element ha passed the specified ranking filter, else nan. Ranking threshold can be
    in percentage terms or a number.
    :param df: DataFrame
    :param ascending: bool
    :param inclusive: bool
    :param rank_threshold: float or int
    :return: DataFrame
    """
    # check the inputs
    if rank_threshold <= 0:
        raise ValueError("'rank_threshold' needs to be an int or float strictly larger than 0")
    elif type(rank_threshold) == float and rank_threshold > 1:
        raise ValueError("if 'rank_threshold' is a float, it needs to be less than or equal to 1")

    # rank the elements in the DataFrame
    ranked_df = df.rank(axis='columns', method='first', ascending=ascending, numeric_only=True)

    # set DataFrame to 1 if the ranking filter is passed, else nan
    if type(rank_threshold) == float:
        num_numeric_per_row = ranked_df.count(axis=1)
        rank_threshold = round(num_numeric_per_row * rank_threshold)
        _filter = np.where(ranked_df.le(rank_threshold, axis=0), inclusive, not inclusive)
    else:
        _filter = np.where(ranked_df <= rank_threshold, inclusive, not inclusive)
    filter_df = pd.DataFrame(index=df.index, columns=df.columns, data=_filter)  # store result in a DataFrame
    filter_df *= 1  # convert True to 1 and False to 0
    filter_df *= np.where(~df.isnull(), 1, np.nan)  # nan in case data is nan
    filter_df.replace(0, np.nan, inplace=True)
    return filter_df


def main():
    from tools.excel_tools import load_df
    df = load_df(full_path=r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\excel_data\dataframe_test.xlsx')
    print(df)

    df_win = winsorize_dataframe(df)
    print(df_win)

    print(rank_filter(df=df_win, ascending=False, inclusive=True, rank_threshold=5.))


if __name__ == '__main__':
    main()

