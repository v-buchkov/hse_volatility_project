from ml_experiments.src.experiments_handler import run_experiments, AvailableInstruments


if __name__ == '__main__':
    RANDOM_STATE = 12

    PATH_TEXTS = 'data/telegram/'
    PATH_PNL = 'data/pnl/'
    PATH_OUTPUT = 'output/'

    BACKTEST_FX = 'USDRUB'
    INSTRUMENT_TYPE = AvailableInstruments.OPTION
    BACKTEST_DAYS = 5
    YEAR = 2022
    USE_ALL_SOURCES = True

    run_experiments(currency=BACKTEST_FX, days_strategy=BACKTEST_DAYS, year=YEAR, texts_path=PATH_TEXTS,
                    target_path=PATH_PNL, output_path=PATH_OUTPUT, instrument_type=INSTRUMENT_TYPE,
                    use_all_sources=USE_ALL_SOURCES, random_state=RANDOM_STATE)
