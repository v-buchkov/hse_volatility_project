import os
import datetime as dt
from typing import List, Tuple, Union

import pandas as pd

from ml_experiments.src.technical.search import binary_search_by_date
from ml_experiments.src.dataset.dataloader import load_texts_df, load_target_variable, convert_target_to_timeseries


def get_list_of_available_sources(data_path: str) -> List[str]:
    return os.listdir(data_path)


def _get_label_by_date(row, target_timeseries: List[Tuple[dt.datetime, int]]) -> int:
    initial_date = target_timeseries[0][0]
    date_x = pd.to_datetime(row['date']).to_pydatetime()
    pnl_sign_key = binary_search_by_date(target_timeseries, date_x)

    if pnl_sign_key is not None and date_x >= initial_date:
        return target_timeseries[pnl_sign_key][1]


def add_target_by_date(texts_df: pd.DataFrame, target_timeseries: List[Tuple[dt.datetime, int]]) -> pd.DataFrame:
    texts_df['label'] = texts_df.apply(lambda row: _get_label_by_date(row, target_timeseries), axis=1)
    texts_df.dropna(subset=['label'], inplace=True)
    texts_df.reset_index(drop=True, inplace=True)
    return texts_df


def compose_initial_dataset(currency: str, days_strategy: int, year: int, texts_path: str, target_path: str,
                            sources_subset: Union[List[str], None] = None) -> pd.DataFrame:
    if sources_subset is None:
        sources_subset = get_list_of_available_sources(texts_path)

    texts = load_texts_df(sources_subset, texts_path)
    target = load_target_variable(currency, days_strategy, year, target_path)

    df = add_target_by_date(texts, convert_target_to_timeseries(target))
    df.drop(['id'], axis=1, inplace=True)

    return df
