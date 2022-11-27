from sortedcontainers import SortedDict

from stat_arb.src.data_loader.general.Interval import Interval


class ScheduleOfChanges(SortedDict):
    """
    ScheduleOfChanges class is a class created for convenient work with the objects of type
    spread_setting_schedule and to localize all code related to logic of work with such objects. \n
    ScheduleOfChanges is based on SortedDict, where
        - key usually refers to time of change;
        - value is a new value (e.g. spread setting) that is valid starting from time specified
        in key.
    In fact, type of the key is not limited to datetime or date, and all the methods can be
    applied to ScheduleOfChanges with non-time keys (e.g. integers).
    """

    def __init__(self, *args, **kwargs):
        super(ScheduleOfChanges, self).__init__(*args, **kwargs)

    def floor_value(self, key, default_floor=None):
        """
        returns the
        """
        floor_key = self.floor_key(key)
        if floor_key is None:
            return default_floor
        return self[floor_key]

    def floor_key(self, key):
        idx = self.bisect_right(key)
        if idx == 0:
            return None
        return self.iloc[idx - 1]

    def interval_to_next(self, key, defaul_next=None):
        return Interval(key, self.next_key(key, defaul_next))

    def next_key(self, key, default_next=None):
        idx = self.bisect_right(key)
        if idx == len(self):
            return default_next
        return self.iloc[idx]

    def get_changes_for_interval(self, interval, none_if_not_available=False):
        start = interval.start
        end = interval.end
        idx_left = self.bisect_right(start)
        if idx_left == 0:
            if not none_if_not_available:
                raise Exception("There is no elements before %s " % str(start))
            first_item = [start, None]
        else:
            first_item = list(self.peekitem(idx_left - 1))
        # set time of the first record to start of the requested interval
        first_item[0] = start
        yield tuple(first_item)
        idx_right = self.bisect_right(end)
        for idx in range(idx_left, idx_right):
            yield self.peekitem(idx)

    def get_splitted_time_intervals_mapping(self, interval):
        only_start_change = list(self.get_changes_for_interval(interval))
        final = []
        for i, value in enumerate(only_start_change):
            start = value[0]
            if i + 1 <= len(only_start_change) - 1:
                end = only_start_change[i + 1][0]
            else:
                end = interval.end
            if start == end:
                continue
            small_int = Interval(start, end)
            final.append((small_int, value[1]))
        return final

    @staticmethod
    def get_schedule_of_changes_factory(records_factory):
        def factory(key):
            records = records_factory(key)
            schedule = ScheduleOfChanges()
            for item in records:
                schedule[item.start_time] = item.spread_setting
            return schedule

        return factory
