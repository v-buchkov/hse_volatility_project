"""Script for generating synthetic FX pair prices from other two replicating FX pairs"""
from static_data import PATH, DELTA_SECONDS
from typing import Dict
import datetime as dt
from stat_arb_old.src.working_with_files.preprocessing import get_asset_prices
from stat_arb_old.src.working_with_files.xml_to_csv import save_to_csv


def get_synthetic_fx(base_asset_prices: Dict[dt.datetime, float],
                     second_asset_prices: Dict[dt.datetime, float]) -> Dict[dt.datetime, float]:
    """
    Creates {timestamp: price} dict of synthetic FX pair.
    For instance, generates CNHRUB from USDRUB (base_asset) and USDCNH (second_asset).

        Parameters:
            base_asset_prices (dict): Base asset (value in the numerator) {timestamp: price} dict.
            second_asset_prices (dict) Second asset (value in the denominator) {timestamp: price} dict.
        Returns:
            synthetic_fx (dict): Synthetic asset {timestamp: price} dict.
    """
    synthetic_fx = {}
    for day, base_ccy_price in base_asset_prices.items():
        if day in second_asset_prices.keys():
            synthetic_fx[day] = base_ccy_price / second_asset_prices[day]
    return synthetic_fx


if __name__ == '__main__':
    # Get necessary FX pairs adat
    cnhrub_moex = get_asset_prices(path=PATH, asset='CNHRUB', price_source='moex_fixing', delta_seconds=DELTA_SECONDS)
    usdrub_moex = get_asset_prices(path=PATH, asset='USDRUB', price_source='moex_fixing', delta_seconds=DELTA_SECONDS)
    usdcnh_offshore = get_asset_prices(path=PATH, asset='USDCNH', price_source='bbg', delta_seconds=DELTA_SECONDS)

    # Calculate synthetic FX
    cnh_rub_synthetic = get_synthetic_fx(base_asset_prices=usdrub_moex, second_asset_prices=usdcnh_offshore)

    # Record synthetic FX dict into csv file
    save_to_csv(input_data=(list(cnh_rub_synthetic.items())), path=PATH, directory='bbg', filename='CNHRUB')
