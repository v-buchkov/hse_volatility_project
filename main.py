from sklearn.metrics import balanced_accuracy_score

from ml_experiments.src.experiments_handler import run_experiments, AvailableInstruments


if __name__ == '__main__':
    RANDOM_STATE = 12

    PATH_TEXTS = 'data/telegram/'
    PATH_PNL = 'data/pnl/'
    PATH_OUTPUT = 'output/'

    BACKTEST_FX = 'EURUSD'
    INSTRUMENT_TYPE = AvailableInstruments.OPTION
    BACKTEST_DAYS = 5
    YEAR = 2022

    SOURCES_SUBSET = ['cbonds.csv', 'mmi.csv', 'bitkogan.csv']
    # can set 'default' to use default metric
    QUALITY_METRIC = balanced_accuracy_score

    run_experiments(currency=BACKTEST_FX, days_strategy=BACKTEST_DAYS, year=YEAR, texts_path=PATH_TEXTS,
                    target_path=PATH_PNL, output_path=PATH_OUTPUT, instrument_type=INSTRUMENT_TYPE,
                    sources_subset=SOURCES_SUBSET, random_state=RANDOM_STATE, quality_metric=QUALITY_METRIC)
