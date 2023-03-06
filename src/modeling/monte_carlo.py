import numpy as np
from typing import List


def geometric_brownian_motion(spot: float, years: float, mean: float, vol: float, n_paths: int = 20000,
                              delta_seconds: int = 9 * 60 * 60) -> List[list]:
    """
    Generates random paths of one asset by simple Geometric Brownian Motion with constant volatility.

        Parameters:
            spot (float): Current spot price
            years (float): Period under analysis
            mean (float): Assumed mean of an asset
            vol (float): Assumed volatility of an asset
            n_paths (float): Number of paths for generation (hyperparameter)
            delta_seconds (int): Change of time in seconds for the simulation (by default daily => 9 * 60 * 60)

        Returns:
            output_paths (list): Generated paths [[path1_spot_t0, path1_spot_t1, ...], [path2_spot_t0, path1_spot_t2, ...], ...]
    """
    intervals = 252 * 9 * 60 * 60 / delta_seconds

    # Calculate intervals for Monte Carlo simulation
    t = int(intervals * years)

    # Create the linear space
    time = np.linspace(0, t / intervals, t)
    # Delta time change in the linear space
    d_time = time[1] - time[0]

    # Constant drift of log-normal, mean return is assumed to be at the level of rates difference
    const_drift = (mean - 0.5 * vol ** 2) * d_time

    output_paths = []
    for k in range(n_paths):
        # Create list of changes vs previous point by formula s(t+1) = s(t) * exp(mu + std.dev * Z)
        change = np.exp(const_drift + vol * np.sqrt(d_time) * np.random.normal(0, 1, size=len(time) - 1))

        # Create the list of spots by converting change into list of cumulative returns * SPOT_START
        path = [spot]
        for i in range(len(change)):
            path.append(path[i] * change[i])

        output_paths.append(path)

    return output_paths
