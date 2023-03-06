"""Function for mathematical operations with volatility / covariance"""
import numpy as np
import scipy.stats
import scipy.optimize
from typing import List


def calculate_vol_implied(opt_object, opt_price: float, **opt_data) -> float:
    """Returns implied vol of specified option object at specified price.

    Function iterates over different vols, calculating option price, and minimizes MSE of difference in price estimation

        Parameters:
            opt_object (class): Non-linear volatility derivative
            opt_price (float): Price of the derivative (in the format the opt_object returns)
            opt_data (kwargs): All required parameters for opt_object initialization, except for volatility

        Returns:
            solution (float): Volatility, implied by the specified price
    """
    # Function of MSE
    def error_estimation(vol: float) -> float:
        opt_data['underlying_volatility'] = vol[0]
        estimated_price = opt_object(**opt_data).price()
        return (estimated_price - opt_price) ** 2

    # Initialized volatility variable
    vol_const = np.array([0.001])

    # Find solution via minimization of error estimation by volatility variable
    solution = scipy.optimize.minimize(error_estimation, vol_const, bounds=[(0.05, 1)]).x[0]

    return solution


def calculate_vol_realized(asset_returns: list, delta_seconds: int) -> float:
    """
    Calculates annualized realized volatility.

    Uses 252 trading days, 8 hours, 60 minutes, 60 seconds convention

        Parameters:
            asset_returns (list): List of returns (log-price difference) for volatility calculation
            delta_seconds (int): Timestamp difference between returns in seconds (e.g., 9 * 60 * 60 for daily spacing)

        Returns:
            annualized_vol (float): Annualized volatility for given asset
    """
    annualized_vol = np.sqrt(252 * 9 * 60 * 60 / delta_seconds) * np.std(asset_returns)
    return annualized_vol


def calculate_cov_implied(vol_implied_asset1: float, vol_implied_asset2: float, vol_replicating: float) -> float:
    """
    Calculates covariance between two assets, implied in the volatilities of these assets
    and third asset that can perfectly replicate volatility of returns of first two assets combination.

        Parameters:
            vol_implied_asset1 (float): Volatility of 1st asset
            vol_implied_asset2 (float): Volatility of 2nd asset
            vol_replicating (float): Volatility of an asset that can perfectly replicate volatility of this combination

        Returns:
            cov_implied (float): Implied covariance
    """
    cov_implied = (vol_implied_asset1 + vol_implied_asset2 - vol_replicating) / 2
    return cov_implied


def calculate_cov_realized(returns_asset1: List[float], returns_asset2: List[float], delta_seconds: int) -> float:
    """
    Calculates realized covariance between two assets.

        Parameters:
            returns_asset1 (list): Returns (log-price difference) of 1st asset
            returns_asset2 (list): Returns (log-price difference) of 2nd asset
            delta_seconds (int): Timestamp difference between returns in seconds (e.g., 9 * 60 * 60 for daily spacing)

        Returns:
            cov_realized (float): Realized covariance
    """
    vol_asset1 = calculate_vol_realized(asset_returns=returns_asset1, delta_seconds=delta_seconds)
    vol_asset2 = calculate_vol_realized(asset_returns=returns_asset2, delta_seconds=delta_seconds)

    cov_realized = scipy.stats.pearsonr(returns_asset1, returns_asset2)[0] * vol_asset1 * vol_asset2
    return cov_realized
