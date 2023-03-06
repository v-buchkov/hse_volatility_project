"""File with static data for backtest"""
PATH = 'data/'
# Change in seconds, for which we aggregate the return (used for HFT backtest)
DELTA_SECONDS = 60 * 60
# Dict[Asset -> Code on MOEX]
ONSHORE_ASSETS = {'EURRUB': 'Eu', 'USDRUB': 'Si', 'EURUSD': 'ED'}
OFFSHORE_ASSETS = {'EURUSD': 'ED', 'USDCNH': 'CNY'}
