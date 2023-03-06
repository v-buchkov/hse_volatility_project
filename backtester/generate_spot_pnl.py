import pandas as pd

if __name__ == '__main__':
    PATH = '../data/'
    PATH_OUT = '../data/pnl/spot/'
    SOURCE = 'moex'
    ASSET = 'CNHRUB'
    DAYS = 5
    YEAR = 2022

    spot_pnl = pd.read_csv(PATH + SOURCE + '/' + ASSET + '.csv')

    spot_pnl.dropna(inplace=True)
    spot_pnl['date_start'] = pd.to_datetime(spot_pnl['timestamp']).dt.strftime('%Y-%m-%d')
    spot_pnl.drop(['timestamp'], axis=1, inplace=True)

    spot_pnl.drop_duplicates(subset=['date_start'], keep='first', inplace=True)
    spot_pnl.reset_index(inplace=True, drop=True)

    spot_pnl['Mid'] = (spot_pnl['Bid'] + spot_pnl['Ask']) / 2

    spot_pnl['pnl'] = spot_pnl['Mid'].div(spot_pnl['Mid'].shift(DAYS)) - 1

    spot_pnl.to_csv(f'{PATH_OUT}Backtest_{ASSET}_{DAYS}_days_{YEAR}.txt', index=False)
