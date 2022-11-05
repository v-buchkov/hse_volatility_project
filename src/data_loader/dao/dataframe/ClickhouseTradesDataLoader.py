import pytz

from stat_arb.src.data_loader.general.Interval import Interval
import pandas as pd

UTC_TIMEZONE = pytz.timezone('UTC')


class ClickhouseTradesDataLoader:
    def __init__(self, connection):
        self.connection = connection

    def load_trades_for_interval(self, provider: str, instrument: str, interval: Interval):
        query = '''
            select * from trade
            where true
                and date_time >= toDateTime64('{start}', 9)
                and date_time < toDateTime64('{end}', 9)
                and provider = '{provider}'
                and instrument = '{instrument}'
                order by date_time asc
        '''.format(start=interval.start.strftime('%Y-%m-%d %H:%M:%S.%f000'),
                   end=interval.end.strftime('%Y-%m-%d %H:%M:%S.%f000'),
                   provider=provider,
                   instrument=instrument)
        df = pd.read_sql(query, self.connection)
        df.rename({'date_time': 'timestamp'}, axis=1, inplace=True)
        df.set_index('timestamp', drop=True, inplace=True)
        df.index = df.index.tz_localize(UTC_TIMEZONE)
        return df


if __name__ == '__main__':
    from datetime import datetime
    from database import database_config
    from general.Interval import Interval

    with database_config.sql_engine_clickhouse_prod_portforward.connect() as connection:
        loader = ClickhouseTradesDataLoader(connection.connection.connection)
        interval = Interval(datetime(2022, 10, 7), datetime(2022, 10, 8))
        df_trades = loader.load_trades_for_interval('MOEX_EQ',
                                                    "GAZP/RUB_TQBR",
                                                    interval)
        print(df_trades)
