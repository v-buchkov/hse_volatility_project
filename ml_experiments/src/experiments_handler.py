import enum
from typing import List, Tuple, Union, Any
from contextlib import redirect_stdout

import pandas as pd
from fast_ml.model_development import train_valid_test_split
from tqdm import tqdm

from ml_experiments.src.dataset.dataset_composer import compose_initial_dataset, get_list_of_available_sources
from ml_experiments.src.technical.combinatorics import get_all_combinations

from ml_experiments.src.text_preprocessing.embeddings_count_vectorizer import count_vectorizer_embedding
from ml_experiments.src.text_preprocessing.embeddings_tfidf import tfidf_embedding
from ml_experiments.src.text_preprocessing.embeddings_word2vec import w2v_embedding
from ml_experiments.src.text_preprocessing.embeddings_word2vec_pretrained import w2v_embedding_pretrained
from ml_experiments.src.text_preprocessing.embeddings_glove_pretrained import glove_embedding_pretrained
from ml_experiments.src.text_preprocessing.embeddings_fasttext import fasttext_embedding

from ml_experiments.src.modeling.logreg import logreg
from ml_experiments.src.modeling.random_forest import random_forest
from ml_experiments.src.modeling.boosting import catboosting
from ml_experiments.src.modeling.fasttext_classifier import fasttext_classifier


class AvailableInstruments(enum.Enum):
    SPOT = 'spot/'
    OPTION = 'option/'


def calculate_sample_balance(target_variable: pd.Series) -> float:
    balance = target_variable.sum() / target_variable.shape[0]
    return max(balance, 1 - balance)


def _logger(preprocesser, model, train_score: float, val_score: float,
            test_score: float):
    print(f'Preprocess: {preprocesser.__name__}, Model: {model.__name__}, '
          f'Train @: {round(train_score, 4)}, Val @: {round(val_score, 4)}, Test @: {round(test_score, 4)}')
    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------\n')


def _run_single_experiment(currency: str, days_strategy: int, year: int, texts_path: str, target_path: str,
                           output_path: str, instrument_type: AvailableInstruments = AvailableInstruments.OPTION,
                           sources_subset: Union[List[str], None] = None, train_size: float = 0.8,
                           random_state: int = 12, quality_metric: Union[Any, str] = 'default',
                           iter_num: Union[int, str] = '') -> Tuple[Tuple, float, float]:
    output_filename = f'{instrument_type.value}single_experiments_logs/ML_Experiments_{currency}_{days_strategy}_days_{year}_sources{iter_num}.txt'

    with open(output_path + output_filename, 'w') as out_f:
        with redirect_stdout(out_f):
            target_path += instrument_type.value
            df = compose_initial_dataset(currency, days_strategy, year, texts_path, target_path, sources_subset)
            X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(df, target='label',
                                                                                    train_size=train_size,
                                                                                    valid_size=(1 - train_size) / 2,
                                                                                    test_size=(1 - train_size) / 2,
                                                                                    random_state=random_state)
            if sources_subset is None:
                print(f'Sources List = all sources')
            else:
                print(f'Sources List = {sources_subset}')

            print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}')
            print(f'Balances: Train = {round(calculate_sample_balance(y_train), 4)}, '
                  f'Val = {round(calculate_sample_balance(y_val), 4)}, '
                  f'Test = {round(calculate_sample_balance(y_test), 4)}')

            preprocessers = [count_vectorizer_embedding, tfidf_embedding, w2v_embedding, w2v_embedding_pretrained,
                             glove_embedding_pretrained, fasttext_embedding]
            models = [logreg, random_forest, catboosting, fasttext_classifier]

            fasttext_classifier_trained = False
            best_pair = (None, None)
            best_val_score = 0
            best_test_score = 0
            for preprocesser in preprocessers:
                if 'pretrained' in preprocesser.__name__ or preprocesser == fasttext_embedding:
                    X_train_p = preprocesser(X_train)
                    X_val_p = preprocesser(X_val)
                    X_test_p = preprocesser(X_test)
                else:
                    X_train_p, X_val_p, X_test_p = preprocesser(X_train, X_val, X_test)

                for model in models:
                    print('New iteration...')

                    if model != fasttext_classifier:
                        train_score, val_score, test_score = model(X_train_p, y_train, X_val_p, y_val, X_test_p, y_test,
                                                                   quality_metric=quality_metric,
                                                                   random_state=random_state)
                    elif fasttext_classifier_trained:
                        continue
                    else:
                        train_score, val_score, test_score = model(X_train, y_train, X_val, y_val, X_test, y_test,
                                                                   quality_metric=quality_metric)
                        fasttext_classifier_trained = True

                    _logger(preprocesser, model, train_score, val_score, test_score)

                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_test_score = test_score

                        best_pair = (preprocesser, model)

            print('Solution:')
            print(f'{best_pair}, {best_val_score}, {best_test_score}')

    return best_pair, best_val_score, best_test_score


def run_experiments(currency: str, days_strategy: int, year: int, texts_path: str, target_path: str,
                    output_path: str, instrument_type: AvailableInstruments = AvailableInstruments.OPTION,
                    sources_subset: Union[bool, List[str]] = True, random_state: int = 12,
                    quality_metric: Union[Any, str] = 'default'):
    output_filename = f'{instrument_type.value}FINAL_EXPERIMENT_{currency}_{days_strategy}_days_{year}.txt'

    with open(output_path + output_filename, 'w') as out_f:
        with redirect_stdout(out_f):
            sources = get_list_of_available_sources(texts_path)

            if type(sources_subset) == bool and sources_subset:
                print(f'Using all {len(sources)} sources: {sources}')

                print('Solution:')
                print(_run_single_experiment(currency=currency, days_strategy=days_strategy, year=year,
                                             texts_path=texts_path, target_path=target_path, output_path=output_path,
                                             instrument_type=instrument_type, sources_subset=sources,
                                             random_state=random_state, quality_metric=quality_metric))

            elif type(sources_subset) == bool and not sources_subset:
                all_combinations = get_all_combinations(sources)
                for sources_subset in all_combinations:
                    print(f'Using {len(sources_subset)} sources: {sources_subset}')

                    print('Solution:')
                    print(_run_single_experiment(currency=currency, days_strategy=days_strategy, year=year,
                                                 texts_path=texts_path, target_path=target_path,
                                                 output_path=output_path, instrument_type=instrument_type,
                                                 quality_metric=quality_metric, sources_subset=sources_subset))
            elif type(sources_subset) == list:
                print(f'Using {len(sources_subset)} sources: {sources_subset}')

                print('Solution:')
                print(_run_single_experiment(currency=currency, days_strategy=days_strategy, year=year,
                                             texts_path=texts_path, target_path=target_path,
                                             output_path=output_path, instrument_type=instrument_type,
                                             quality_metric=quality_metric, sources_subset=sources_subset))
            else:
                raise AssertionError(f'Unknown sources_subset variable behaviour!')
