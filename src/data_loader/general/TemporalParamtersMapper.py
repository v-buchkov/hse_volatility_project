from datetime import date, datetime

import pandas as pd

from stat_arb.src.data_loader.general.Interval import Interval
from stat_arb.src.data_loader.general.ScheduleOfChanges import ScheduleOfChanges


class TemporalParametersMapper:
    def __init__(self, start_datetimes_params: dict):
        self._initial_params = ScheduleOfChanges(start_datetimes_params)
        self._intervals, self._params = self.__init_intevals(start_datetimes_params)
        self._n_periods = len(self._intervals)

    def get_working_params_by_date_params(self, timestamp):
        params = self._initial_params.floor_value(timestamp)
        return params

    def __init_intevals(self, start_datetimes_params):
        start_datetimes_sorted = sorted(start_datetimes_params.keys())
        params = [start_datetimes_params[x] for x in start_datetimes_sorted]

        for i in range(len(start_datetimes_sorted)):
            if isinstance(start_datetimes_sorted[i], date):
                start_datetimes_sorted[i] = datetime.combine(start_datetimes_sorted[i],
                                                             datetime.min.time())
        intervals = []
        last_datetime = start_datetimes_sorted[0]
        for i in range(len(start_datetimes_sorted) - 1):
            current_datetime = start_datetimes_sorted[i + 1]
            intervals.append(Interval(last_datetime, current_datetime))
            last_datetime = current_datetime
        intervals.append(Interval(last_datetime, datetime(2262, 4, 11)))
        assert len(intervals) == len(params)
        return intervals, params

    def __getitem__(self, item):
        for i in range(self._n_periods):
            if item in self._intervals[i]:
                return self._params[i]
        raise KeyError(f"{item} not in any period")

    def get_timeinterval_split_for_params(self, time_interval):
        time_interval.localize_none()
        return self._initial_params.get_splitted_time_intervals_mapping(time_interval)

    def split_pandas_object_by_intervals(self, df: pd.DataFrame):
        dataframes = []
        params = []
        for i in range(self._n_periods):
            current_df_slice = df.loc[self._intervals[i].start:
                                      self._intervals[i].end]
            if current_df_slice.empty:
                continue
            dataframes.append(current_df_slice)
            params.append(self._params[i])
        return dataframes, params
