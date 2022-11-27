import logging
from enum import Enum

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool

from stat_arb.src.data_loader.database.CredentialLoader import CredentialLoader

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# root declarative base for project ORM
Base = declarative_base()


class PgDataBase(Enum):
    TRADING = "postgres_trading"
    FXET = "postgres_fxet"
    FXET_DB1 = "postgres_fxet_legacy"
    DB_OPERATIONS = "postgres_db_operations"
    FXONLINE = "postgres_fxonline"

    # Useful when accessing from mac
    # Port forward example:
    # while true; do ssh -L 5432:s-msk-p-fxr-db3.raiffeisen.ru:5432 fxet-centos; sleep 3s; done;
    FXET_PORTFORWARD = "postgres_fxet_portforward"
    FXET_DB1_PORTFORWARD = "postgres_fxet_legacy_portforward"


class ClickhouseServer(Enum):
    PROD_READER = "clickhouse_prod"
    PROD_WRITER = "clickhouse_prod_writer"
    PREVIEW_READER = "clickhouse_preview_reader"
    PREVIEW_WRITER = "clickhouse_preview_writer"
    TEST_READER = "clickhouse_test_reader"
    TEST_WRITER = "clickhouse_test_writer"

    # Useful when accessing from mac
    # Port forward example:
    # while true; do ssh -L 9000:s-msk-p-fxr-cs9:9000 fxet-centos; sleep 3s; done;
    PROD_READER_PORTFORWARD = "clickhouse_prod_portforward"


def create_postgres_engine(login: str, password: str, host: str, port: int, dbname: str) -> Engine:
    return create_engine(
        'postgresql://{}:{}@{}:{}/{}'.format(login, password, host, str(port), dbname)
        + '?keepalives_idle=4&keepalives_interval=5&keepalives_count=5',
        echo=True,
        pool_pre_ping=True,
        pool_recycle=3600,
        execution_options=dict(stream_results=True),
        server_side_cursors=True,
        connect_args={"options": "-c timezone=utc"},
        poolclass=NullPool)


def create_clickhouse_engine(login: str, password: str, host: str, port: int, dbname: str) -> Engine:
    return create_engine(
        'clickhouse+native://{}:{}@{}:{}/{}'.format(login, password, host, str(port), dbname)
        # + '?keepalives_idle=4&keepalives_interval=5&keepalives_count=5',
        # echo=True,
        # pool_pre_ping=True,
        # pool_recycle=3600,
        # execution_options=dict(stream_results=True),
        # server_side_cursors=True,
        # connect_args={"options": "-c timezone=utc"},
        # poolclass=NullPool
    )


def postgres_engine_from_db_config(config_name: str):
    credentials = CredentialLoader.load_credentials(config_name)
    if not credentials:
        print("Can't load config %s", config_name)
        return None

    return create_postgres_engine(credentials.login,
                                  credentials.password,
                                  credentials.host,
                                  credentials.port,
                                  credentials.database)


def clickhouse_engine_from_db_config(config_name: str):
    credentials = CredentialLoader.load_credentials(config_name)
    if not credentials:
        print("Can't load config %s", config_name)
        return None

    return create_clickhouse_engine(credentials.login,
                                    credentials.password,
                                    credentials.host,
                                    credentials.port,
                                    credentials.database)


# sql_engine_trading = postgres_engine_from_db_config(PgDataBase.TRADING.value)
# sql_engine_fxet = postgres_engine_from_db_config(PgDataBase.FXET.value)
# sql_engine_fxet_portforward = postgres_engine_from_db_config(PgDataBase.FXET_PORTFORWARD.value)
sql_engine_fxet_db1 = postgres_engine_from_db_config(PgDataBase.FXET_DB1.value)
# sql_engine_fxet_db1_portforward = postgres_engine_from_db_config(PgDataBase.FXET_DB1_PORTFORWARD.value)
# db_operations_engine = postgres_engine_from_db_config(PgDataBase.DB_OPERATIONS.value)
# db_operations_cm_engine = postgres_engine_from_db_config(PgDataBase.DB_OPERATIONS.value)
# db_operations_fxo_engine = postgres_engine_from_db_config(PgDataBase.FXONLINE.value)

# sql_engine_clickhouse_prod = clickhouse_engine_from_db_config(ClickhouseServer.PROD_READER.value)
# sql_engine_clickhouse_prod_portforward = clickhouse_engine_from_db_config(ClickhouseServer.PROD_READER_PORTFORWARD.value)
# sql_engine_clickhouse_prod_writer = clickhouse_engine_from_db_config(ClickhouseServer.PROD_WRITER.value)
# sql_engine_clickhouse_preview_writer = clickhouse_engine_from_db_config(ClickhouseServer.PREVIEW_WRITER.value)
# sql_engine_clickhouse_preview_reader = clickhouse_engine_from_db_config(ClickhouseServer.PREVIEW_READER.value)
# sql_engine_clickhouse_test_writer = clickhouse_engine_from_db_config(ClickhouseServer.TEST_WRITER.value)
# sql_engine_clickhouse_test_reader = clickhouse_engine_from_db_config(ClickhouseServer.TEST_READER.value)
