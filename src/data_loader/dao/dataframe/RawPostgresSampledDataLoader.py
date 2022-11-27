from datetime import datetime, timedelta
import io
from enum import Enum

import lz4.frame
import numpy as np
import pandas as pd
import pytz
from stat_arb.src.data_loader.general.Interval import Interval

from stat_arb.src.data_loader.general.TemporalParamtersMapper import TemporalParametersMapper

UTC_TIMEZONE = pytz.timezone('UTC')


class BinaryPostgresTypes(Enum):
    NUMPY = 'numpy'
    LZ4_NUMPY = 'lz4_numpy'


class ArtificialInstrument:
    def __init__(self, ccy_pair, tenor):
        self.currency_pair = ccy_pair
        self.tenor = tenor

    def to_bson_string(self) -> str:
        return '{}_T+{}'.format(self.currency_pair, self.tenor)


VWAP_PROVDERS_MAPPER = TemporalParametersMapper(
    {datetime(1970, 1, 1): {'REUTERS_MATCHING': 'REUTERS_MATCHING'},
     datetime(2021, 5, 1): {'REUTERS_MATCHING':
                                'LND_HARDS_FOR_RETURNS'},
     datetime(2022, 3, 2): {'REUTERS_MATCHING':
                                'LND_HARDS_FOR_RETURNS'}})

VWAP_BY_INSTRUMENT_PROVDERS_MAPPER = {'USD/CNH':
                                          TemporalParametersMapper({datetime(1970, 1, 1): {'EBS': 'EBS'},
                                                                    datetime(2020, 1, 1): {'EBS':
                                                                                               'LND_HARDS_FOR_RETURNS'}})}

PROVIDER_INSTRUMENT_MAPPER = {('MOEX', 'CNH/RUB_TOM'): ('MOEX', ArtificialInstrument('CNY/RUB', 1)),
                              ('MOEX', 'USD/CNH_TOM'): ('MOEX', ArtificialInstrument('USD/CNY', 1))}


class RawPostgresSampledDataLoader:
    VWAP_TABLE_NAME = 'vwaps'

    provider_dict = {}

    def __init__(self, postgres_connection):
        self.db = postgres_connection

    @classmethod
    def transform_source(cls, source):
        try:
            return cls.provider_dict[str(source)]
        except KeyError:
            return [str(source)]

    @classmethod
    def transform_size_for_source(cls, source, size):
        if str(source) == 'MOEX_BOND' or str(source) == 'BPIPE_MICX':
            return size / 1000
        return size

    @classmethod
    def size_by_spread_keys(cls, instrument, sampling_schema, source, spread):
        sources = RawPostgresSampledDataLoader.transform_source(source)
        size = RawPostgresSampledDataLoader.transform_size_for_source(source, spread)
        return [("SIZE_BY_SPREAD-%s-%s-%s-%4.3f" %
                 (instrument,
                  sampling_schema.name,
                  source,
                  size)) for source in sources]

    @classmethod
    def vwap_keys(cls, instrument, sampling_schema, source, size, is_best_price=False):
        sources = RawPostgresSampledDataLoader.transform_source(source)
        size = RawPostgresSampledDataLoader.transform_size_for_source(source, size)
        vwap_name = "VWAP_L1" if is_best_price else "VWAP"
        return [("%s-%s-%s-%s-%d" %
                 (vwap_name,
                  instrument,
                  sampling_schema.name,
                  source,
                  size)) for source in sources]

    @classmethod
    def ob_state_keys(cls, instrument, sampling_schema, source):
        sources = RawPostgresSampledDataLoader.transform_source(source)
        return [("OB_STATE-%s-%s-%s" %
                 (instrument,
                  sampling_schema.name,
                  source)) for source in sources]

    @classmethod
    def ob_simple_keys(cls, instrument, sampling_schema, source):
        sources = RawPostgresSampledDataLoader.transform_source(source)
        return [("SIMPLE_OB-%s-%s-%s" %
                 (instrument,
                  sampling_schema.name,
                  source)) for source in sources]

    @classmethod
    def raw_quotes_keys(cls, instrument, source):
        sources = RawPostgresSampledDataLoader.transform_source(source)
        return [("RAW_QUOTES-%s-%s" %
                 (instrument,
                  source)) for source in sources]

    @staticmethod
    def convert_str_date_to_datetime(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date()

    @classmethod
    def derivatives_features_key(cls,
                                 currency,
                                 rate_type,
                                 floating_index,
                                 additional=None):
        return [("%s|%s|%s|%s" % (currency, rate_type, floating_index, additional))
                if additional is not None else
                ("%s|%s|%s" % (currency, rate_type, floating_index))]

    @staticmethod
    def decompress_from_stream(input_stream):
        # input_stream.seek(0, 2)
        # print(f'compressed size: {input_stream.tell() / 1024 / 1024} MBytes')
        # input_stream.seek(0)

        # start = time.time()
        output_stream = lz4.frame.decompress(input_stream.read())
        output_stream = io.BytesIO(output_stream[-128:] + output_stream[:-128])
        # print(f"time taken to decompress: {time.time() - start} seconds")
        return output_stream

    def load_data_numpy(self,
                        key,
                        dates,
                        table_name):
        # start = time.time()
        cursor = self.db.cursor()
        query = f'''select metadata, data, start_datetime, format from analysis.{table_name} 
                                WHERE KEY in {str(tuple(key)).replace(",)", ")")} 
                                and start_datetime >= '{min(dates)}'
                                and start_datetime <= '{max(dates)}'
                                ORDER BY start_datetime'''
        cursor.execute(query)

        result_rows = cursor.fetchall()
        # print(f"time taken to load vwaps table {time.time() - start} seconds\n")
        if result_rows is None:
            return None
        dfs = []
        columns = []
        for row in result_rows:
            metadata = row[0]
            columns = metadata['columns']
            binary_data = row[1]
            date = row[2].date()
            format = row[3]
            if date not in dates:
                continue

            # start = time.time()
            lo_object = self.db.lobject(binary_data, 'rb')
            # lo_object = io.BytesIO(lo_object.read())
            # print(f"time to read from postgres: {time.time() - start} seconds")

            # print(f"data format: {row[6]}")

            if format == BinaryPostgresTypes.LZ4_NUMPY.value:
                lo_object = self.decompress_from_stream(lo_object)

            # lo_object.seek(0, 2)
            # print(f'numpy binary size: {lo_object.tell() / 1024 / 1024} MBytes')
            # lo_object.seek(0)

            # start = time.time()
            loaded_values = np.load(lo_object)
            # print(f"time to create df from np binary {time.time() - start} seconds")
            dfs.append(loaded_values)

        if not dfs:
            return None

        # print()
        df = np.concatenate(dfs, axis=0)
        return df, columns

    def load_data(self,
                  key,
                  dates,
                  table_name):
        df, columns = self.load_data_numpy(key, dates, table_name)
        df = df.reshape([-1, len(columns)])
        df = pd.DataFrame(data=df[:, 1:],
                          index=np.array(df[:, 0], dtype='datetime64[us]'),
                          columns=columns[1:])

        df.index.name = 'timestamp'
        df.index = df.index.tz_localize(UTC_TIMEZONE)

        return df

    def load_vwap(self,
                  source,
                  instrument,
                  dates,
                  schema,
                  size,
                  is_best_price=False):
        time_interval = Interval(pd.to_datetime(min(dates)),
                                 pd.to_datetime(max(dates)) + timedelta(days=1))
        if str(instrument) in VWAP_BY_INSTRUMENT_PROVDERS_MAPPER.keys():
            changes_for_time_interval = VWAP_BY_INSTRUMENT_PROVDERS_MAPPER[str(instrument)] \
                .get_timeinterval_split_for_params(time_interval)
        else:
            changes_for_time_interval = VWAP_PROVDERS_MAPPER.get_timeinterval_split_for_params(time_interval)
        all_dat = []
        for small_interval_mapper in changes_for_time_interval:
            small_interval = small_interval_mapper[0]
            if small_interval.empty():
                continue
            mapper = small_interval_mapper[1]
            if str(source) in mapper.keys():
                source = mapper[str(source)]
            if (str(source), str(instrument)) in PROVIDER_INSTRUMENT_MAPPER:
                source, instrument = PROVIDER_INSTRUMENT_MAPPER[(str(source), str(instrument))]
            key = RawPostgresSampledDataLoader.vwap_keys(instrument,
                                                         schema,
                                                         source,
                                                         size,
                                                         is_best_price)
            table_name = self.VWAP_TABLE_NAME
            local_dates = [date for date in dates if date in small_interval]
            dat_by_small_interval = self.load_data(key,
                                                   local_dates,
                                                   table_name)
            all_dat.append(dat_by_small_interval)
        return pd.concat(all_dat, axis=0, copy=False)

    def load_size_by_spread(self,
                            source,
                            instrument,
                            dates,
                            schema,
                            spread):
        key = RawPostgresSampledDataLoader.size_by_spread_keys(instrument,
                                                               schema,
                                                               source,
                                                               spread)
        return self.load_data(key,
                              dates,
                              self.VWAP_TABLE_NAME)

    def load_vwap_for_interval(self, source, instrument, interval, schema, size, is_best_price=False):
        dates = interval.generate_nonweekend_dates_range()
        dates = list(map(lambda x: x.date(), dates))
        return self.load_vwap(source, instrument, dates, schema, size, is_best_price)

    def load_ob_state(self,
                      source,
                      instrument,
                      dates,
                      schema):
        key = RawPostgresSampledDataLoader.ob_state_keys(instrument,
                                                         schema,
                                                         source)
        return self.load_data(key,
                              dates,
                              self.VWAP_TABLE_NAME)

    def load_ob_state_for_interval(self, source, instrument, interval, schema):
        dates = interval.generate_nonweekend_dates_range()
        dates = list(map(lambda x: x.date(), dates))
        return self.load_ob_state(source, instrument, dates, schema)

    def load_simple_ob(self,
                       source,
                       instrument,
                       dates,
                       schema):
        key = RawPostgresSampledDataLoader.ob_simple_keys(instrument, schema, source)
        return self.load_data(key, dates, self.VWAP_TABLE_NAME)

    def load_simple_ob_for_interval(self, source, instrument, interval, schema):
        dates = interval.generate_nonweekend_dates_range()
        dates = list(map(lambda x: x.date(), dates))
        return self.load_simple_ob(source, instrument, dates, schema)

    def load_raw_quotes(self,
                        source,
                        instrument,
                        dates):
        key = RawPostgresSampledDataLoader.raw_quotes_keys(instrument, source)
        return self.load_data(key, dates, self.VWAP_TABLE_NAME)

    def load_raw_quotes_numpy(self,
                              source,
                              instrument,
                              dates):
        key = RawPostgresSampledDataLoader.raw_quotes_keys(instrument, source)
        return self.load_data_numpy(key, dates, self.VWAP_TABLE_NAME)

    def load_raw_quotes_for_interval(self, source, instrument, interval):
        dates = interval.generate_nonweekend_dates_range()
        dates = list(map(lambda x: x.date(), dates))
        return self.load_raw_quotes(source, instrument, dates)

    def load_raw_quotes_numpy_for_interval(self, source, instrument, interval):
        dates = interval.generate_nonweekend_dates_range()
        dates = list(map(lambda x: x.date(), dates))
        return self.load_raw_quotes_numpy(source, instrument, dates)


# if __name__ == '__main__':
#     from datetime import datetime
#     from stat_arb.src.data_loader.database.database_config import sql_engine_fxet
#     from stat_arb.src.data_loader.general.Interval import Interval
#     from stat_arb.src.data_loader.general.SamplingSchemas import SamplingSchemas
#
#     with sql_engine_fxet.connect() as connection:
#         loader = RawPostgresSampledDataLoader(postgres_connection=connection.connection.connection)
#
#         interval = Interval(datetime(2019, 10, 14), datetime(2019, 10, 15))
#         df = loader.load_vwap_for_interval('MOEX_DIRECT',
#                                            "USD/RUB_T+1",
#                                            interval,
#                                            SamplingSchemas.FIRST_PRICE_PREDICTION_SCHEMA,
#                                            500_000)
#         print(df)
