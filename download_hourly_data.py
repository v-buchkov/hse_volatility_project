from stat_arb.src.data_loader.dao.dataframe.RawPostgresSampledDataLoader import RawPostgresSampledDataLoader
from stat_arb.src.data_loader.dao.dataframe.ClickhouseTradesDataLoader import ClickhouseTradesDataLoader
from stat_arb.src.data_loader.database import database_config
from datetime import datetime
from stat_arb.src.data_loader.general.Interval import Interval
from stat_arb.src.data_loader.general.SamplingSchemas import SamplingSchemas
from static_data import PATH

queries = [
    {'source': 'MOEX_DIRECT', 'instrument': 'USD/RUB_T+1', 'size': 1_000_000},
    {'source': 'MOEX_DIRECT', 'instrument': 'EUR/USD_T+1', 'size': 1_000_000},
    {'source': 'MOEX_DIRECT', 'instrument': 'CNH/RUB_T+1', 'size': 1_000_000},
    {'source': 'RBI',         'instrument': 'EUR/USD_T+2', 'size': 1_000_000},
    {'source': 'RBI',         'instrument': 'USD/CNH_T+2', 'size': 1_000_000},
]

# queries = [
#     {'source': 'MOEX_DIRECT', 'instrument': 'CNH/RUB_T+1', 'size': 1_000_000}
# ]

interval = Interval(datetime(2022, 3, 1), datetime(2022, 10, 25))


def load_data(query: dict, interval: Interval):
    print('loading:\n', query, '\n', interval, '\n')
    with database_config.sql_engine_fxet_db1.connect() as connection:
        loader = RawPostgresSampledDataLoader(connection.connection.connection)
        vwap = loader.load_vwap_for_interval(query['source'],
                                             query['instrument'],
                                             interval,
                                             SamplingSchemas.HOURLY_SCHEMA,
                                             query['size'])
    return vwap


if __name__ == '__main__':
    for q in queries:
        source = q['source'].split('_')[0].lower()
        instrument = q['instrument'].split('_')[0].replace('/', '').upper()
        spot_data = load_data(q, interval)
        spot_data.to_csv(f'../{PATH}/{source}/{instrument}.csv')
