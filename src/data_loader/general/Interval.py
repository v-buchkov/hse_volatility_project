from copy import copy
from datetime import timedelta, datetime, date

from pytz import timezone

UTC_TIMEZONE = timezone('UTC')

class Interval(object):
    """
    Represents an interval.
    Defined as half-open interval [start,end), which includes the start position but not the end.
    Start and end do not have to be numeric types (datetime type is supposed).
    """

    def __init__(self, start: datetime, end: datetime):
        if end < start:
            raise ValueError("Start is later than end for the interval [{},{})".format(start, end))
        if end == start:
            raise ValueError("Start==end for the interval [{},{})".format(start, end))
        self._start = start
        self._end = end

    def localize_utc(self):
        if self.start.tzinfo is None:
            self._start = UTC_TIMEZONE.localize(self._start)
        if self.end.tzinfo is None:
            self._end = UTC_TIMEZONE.localize(self._end)

    def localize_none(self):
        self._start = self._start.replace(tzinfo=None)
        self._end = self._end.replace(tzinfo=None)

    def non_localized_copy(self):
        cpy = copy(self)
        cpy.localize_none()
        return cpy

    @property
    def start(self):
        """
        The interval's start.
        """
        return self._start

    @property
    def end(self):
        """
        The interval's end.
        """
        return self._end

    def generate_dates_range(self):
        return [self.start + timedelta(days=x) for x in range(0, (self.end - self.start).days)]

    def split(self, days_step):
        splitted = []
        total_days = (self.end - self.start).days
        start_time = self.start
        end_time = self.end
        for step in range(int(total_days / days_step) + 1):
            timedelta_step = timedelta(days=days_step)
            local_time_interval = Interval(start_time,
                                           start_time + timedelta_step)
            if local_time_interval.end > end_time:
                local_time_interval = Interval(start_time,
                                               end_time)
            splitted.append(local_time_interval)
            start_time += timedelta_step
            # stop if we are already past the end
            if start_time >= end_time:
                break
        return splitted

    def generate_nonweekend_dates_range(self):
        return [self.start + timedelta(days=x) for x in range(0, (self.end - self.start).days)
                if (self.start + timedelta(days=x)).weekday() not in (5, 6)]

    def generate_weekend_dates_range(self):
        return [self.start + timedelta(days=x) for x in range(0, (self.end - self.start).days)
                if (self.start + timedelta(days=x)).weekday() in (5, 6)]

    def generate_nonweekend_dates_range_utc(self):
        date_range = self.generate_nonweekend_dates_range()
        return [datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone('UTC')) for d in date_range]

    def generate_dates_range_for_weekly_rebalancing(self, weekday):
        return [self.start + timedelta(days=x) for x in range(0, (self.end - self.start).days)
                if (self.start + timedelta(days=x)).weekday() == weekday]

    def generate_every_nth_trading_date_range(self, n, instrument):
        return self.generate_nonweekend_noholidays_dates_range(instrument)[::n]

    def generate_nonweekend_noholidays_dates_range(self, instrument):
        date_range = self.generate_nonweekend_dates_range()
        return [datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone('UTC'))
                for d in date_range if d.date() not in list(instrument.currency_pair.left_right_holidays)]

    def generate_nonweekend_holidays_dates_range(self, instrument):
        return [d for d in sorted(list(instrument.currency_pair.usd_holidays))
                if self.start.date() <= d < self.end.date()]

    def generate_yearly_sub_intervals(self):
        """This function will return a list of Intervals, each within a year. Timezone remains the same!"""
        years_in_between = list(range(self.start.year + 1, self.end.year + 1))
        intervals = []
        last_datetime = self.start

        for curr_datetime in list(map(lambda x: datetime(x, 1, 1), years_in_between)) + [self.end]:
            curr_datetime = curr_datetime.replace(tzinfo=self.start.tzinfo)
            if curr_datetime > last_datetime:
                intervals.append(Interval(last_datetime, curr_datetime))
                last_datetime = curr_datetime
        return intervals

    def generate_monthly_sub_intervals(self):
        """This function will return a list of Intervals, each within a month. Timezone remains the same!"""
        if (self.start.year == self.end.year) and (self.start.month == self.end.month):
            # Corner case: sampling less than a month
            return [self]
        res_list = []
        curr_year = self.start.year
        curr_date = self.start
        while curr_year <= self.end.year:
            # 1 == Jan., 12 == Dec.
            start_subset_m = self.start.month + 1 if (curr_year == self.start.year) else 1
            end_subset_m = 12 if (curr_year != self.end.year) else self.end.month
            se_dates_per_year = list(map(lambda m: datetime(curr_year, m, 1, tzinfo=self.start.tzinfo),
                                         list(range(start_subset_m, end_subset_m + 1))))

            for next_date in se_dates_per_year:
                res_list.append(Interval(curr_date, next_date))
                curr_date = next_date

            curr_year += 1
        # The very last month has to contain non-singular interval
        if curr_date < self.end:
            res_list.append(Interval(curr_date, self.end))

        return res_list

    def __str__(self):
        """As string."""
        return '[%s,%s)' % (self.start, self.end)

    def to_string_dates(self):
        return '[%s,%s)' % (self.start.date(), self.end.date())

    def __repr__(self):
        """String representation."""
        return '[%s,%s)' % (self.start, self.end)

    def __cmp__(self, other):
        """Compare."""
        if other is None:
            return 1
        start_cmp = ((self.start > other.start) - (self.start < other.start))
        if 0 != start_cmp:
            return start_cmp
        else:
            return (self.end > other.end) - (self.end < other.end)

    def __lt__(self, other):
        """Less than."""
        if not isinstance(other, Interval):
            return False
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        """Greater than."""
        if not isinstance(other, Interval):
            return False
        return self.__cmp__(other) > 0

    def __le__(self, other):
        """Less than or equal to."""
        if not isinstance(other, Interval):
            return False
        return self.__cmp__(other) <= 0

    def __ge__(self, other):
        """Greater than or equal to."""
        if not isinstance(other, Interval):
            return False
        return self.__cmp__(other) >= 0

    def __eq__(self, other):
        """Equal to."""
        if not isinstance(other, Interval):
            return False
        return (self.start == other.start) and (self.end == other.end)

    def __ne__(self, other):
        """Not equal to."""
        if not isinstance(other, Interval):
            return False
        return (self.start != other.start) or (self.end != other.end)

    def __hash__(self):
        """Hash."""
        return hash(self.start) ^ hash(self.end)

    def intersection(self, other):
        """Intersection. @return: An empty intersection if there is none."""
        self_copy = self
        if self_copy > other:
            other, self_copy = self_copy, other
        if self_copy.end <= other.start:
            return Interval(self_copy.start, self_copy.start)
        return Interval(other.start, self_copy.end if self_copy.end < other.end else other.end)

    def hull(self, other):
        """@return: Interval containing both self and other."""
        self_copy = self
        if self_copy > other:
            other, self_copy = self_copy, other
        return Interval(self_copy.start, other.end)

    def overlap(self, other):
        """@return: True iff self intersects other."""
        self_copy = self
        if self_copy > other:
            other, self_copy = self_copy, other
        return self_copy.end > other.start

    def __contains__(self, timestamp: datetime):
        if isinstance(timestamp, datetime):
            timestamp = timestamp.replace(tzinfo=self._start.tzinfo)
        if not isinstance(timestamp, date):
            raise TypeError(f'argument should be of datetime-like type')
        if isinstance(self._start, datetime) and not isinstance(timestamp, datetime):
            if isinstance(timestamp, date):
                timestamp = datetime.combine(timestamp, datetime.min.time()).replace(tzinfo=self.start.tzinfo)
        elif isinstance(self._start, datetime) and isinstance(timestamp, datetime):
            timestamp = timestamp.replace(tzinfo=self._start.tzinfo)
        return (timestamp >= self.start) and (timestamp < self.end)

    def zero_in(self):
        """@return: True iff 0 in self."""
        return self.start <= 0 < self.end

    def subset(self, other):
        """@return: True iff self is subset of other."""
        return self.start >= other.start and self.end <= other.end

    def proper_subset(self, other):
        """@return: True iff self is proper subset of other."""
        return self.start > other.start and self.end < other.end

    def empty(self):
        """@return: True iff self is empty."""
        return self.start == self.end

    def singleton(self):
        """@return: True iff self.end - self.start == 1."""
        return self.end - self.start == 1

    def separation(self, other):
        """@return: The distance between self and other."""
        self_copy = self
        if self_copy > other:
            other, self_copy = self_copy, other
        if self_copy.end > other.start:
            return 0
        else:
            return other.start - self_copy.end

    def apply(self, function):
        return Interval(function(self.start), function(self.end))
