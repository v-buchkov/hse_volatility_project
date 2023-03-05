import datetime as dt
from typing import List, Tuple, Union


def binary_search_by_date(array: List[Tuple[dt.datetime, float]], date_x: dt.datetime) -> Union[int, None]:
    """
    Searches for the index of date_x in the array via binary search.

        Parameters:
            array (list) : A sorted array of (date, float_value) tuples
            date_x (datetime.datetime) : Date to search for

        Returns:
            index_x (int): Index of the searched date in the array.
    """
    left = 0
    right = len(array) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if array[mid - 1][0] <= date_x <= array[mid][0]:
            return mid
        elif date_x > array[mid - 1][0] and date_x > array[mid][0]:
            left = mid + 1
        else:
            right = mid - 1

    return None


def binary_search_time_series(time_series: List[Tuple[dt.datetime, float]], date_start: dt.datetime,
                              date_end: dt.datetime) -> Union[List[Tuple[dt.datetime, float]], None]:
    """
    Searches for the part of the time series that is contained inside [date_start; date_end] period via binary search.

        Parameters:
            time_series (list) : A sorted array of (date, float_value) tuples
            date_start (datetime.datetime) : Starting date of the searched period
            date_end (datetime.datetime) : Ending date of the searched period

        Returns:
            time_series_data (list): Part of the time series that is contained inside [date_start; date_end] period.
    """

    if date_start <= date_end:
        left_index = _binary_search_by_date(time_series, date_start)

        if left_index is None:
            return None

        right_index = _binary_search_by_date(time_series[left_index:], date_end)

        if right_index is None:
            return None

        right_index += left_index
    else:
        left_index = _binary_search_by_date(time_series, date_end)

        if left_index is None:
            return None

        right_index = _binary_search_by_date(time_series[left_index:], date_start)

        if right_index is None:
            return None

        right_index += left_index

    return time_series[left_index:right_index]
