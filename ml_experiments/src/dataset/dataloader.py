import datetime as dt
from typing import List, Tuple

import pandas as pd

from src.persistence.google_drive import get_file_by_path


def load_texts_df(sources: List[str], data_path: str) -> pd.DataFrame:
    # Generate dataframes
    df = pd.DataFrame()

    for s in sources:
        source_data = pd.read_csv(get_file_by_path(f'{data_path}/{s}'))
        source_data['source'] = s.split('.')[0]
        df = df.append(source_data)

    return df


def load_target_variable(currency: str, days_strategy: int, year_data: int, data_path: str) -> pd.DataFrame:
    # Create target variable dataframe
    target = pd.read_csv(get_file_by_path(f'{data_path}Backtest_{currency}_{days_strategy}_days_{year_data}.txt'))
    target['date_start'] = pd.to_datetime(target['date_start']).dt.strftime('%Y-%m-%d')
    target['label'] = target['pnl'].apply(lambda x: 1 if x >= 0 else 0).astype(int)
    assert target.shape[0] > 0, f'Empty target variable dataset!'
    return target


def convert_target_to_timeseries(target_df: pd.DataFrame) -> List[Tuple[dt.datetime, int]]:
    ts = [(pd.to_datetime(row['date_start']).to_pydatetime(), row['label']) for _, row in target_df.iterrows()]
    return ts
