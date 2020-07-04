import pandas as pd


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
    weights_left_on_col_name = list(right_df)[0]
    result = pd.merge_asof(result, right_df, left_on=index_left_on_col_name, right_on=weights_left_on_col_name,
                           suffixes=('', left_suffix))
    result.set_index(list(result)[0], inplace=True)
    return result



