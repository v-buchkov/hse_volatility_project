import datetime as dt

from backtester import BacktesterDeltaHedgePnL

if __name__ == '__main__':

    fx_pair = 'USDRUB'
    days_strat = [5, 10, 20, 30]
    start_test = dt.datetime(year=2022, month=3, day=1)
    end_test = dt.datetime(year=2022, month=10, day=10)
    fixed_vol = False

    backtest = BacktesterDeltaHedgePnL(asset=fx_pair, rf_base_ccy=0.004, rf_second_ccy=0.025,
                                       datetime_start=start_test, datetime_end=end_test,
                                       onshore_spread=0, offshore_spread=0)

    for days in days_strat:

        if fixed_vol:
            file_name = f'output/Backtest_{fx_pair}_{days}_days_FixedVol.txt'
        else:
            file_name = f'output/Backtest_{fx_pair}_{days}_days.txt'

        backtest.backtest(days_strategy=days, use_fixed_vol=fixed_vol)

        with open(file_name, 'w') as out_f:

            assert len(backtest.pnl_distribution_by_trades) == len(backtest.backtesting_dates)

            out_f.write('date_start,pnl\n')

            for i in range(len(backtest.pnl_distribution_by_trades)):
                out_f.write(f'{backtest.backtesting_dates[i]},{backtest.pnl_distribution_by_trades[i]}\n')

        print(f'{days} days done!')
