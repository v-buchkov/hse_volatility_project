"""Correlation trade strategy files"""


def strategy_decision(replication_vol: float, implied_vol1: float, implied_vol2: float,
                      covariance_predict: float) -> str:
    """
    Returns the decision ('B' or 'S') for 2 delta-neutral option strategies (e.g., MOEX delta(t0)-hedged staddles).

        Parameters:
            replication_vol (float): Volatility of an asset that replicates covariance-adjusted combination of other two
            implied_vol1 (float): Implied volatility of 1st asset
            implied_vol2 (float): Implied volatility of 2nd asset
            covariance_predict (float): Predicted covariance between two assets (e.g., realized covariance)

        Returns:
            decision (str): Decision ('B' - Buy or 'S' - Sell)
    """
    # Calculate implied vol of option strategy (e.g., MOEX Var[EURRUB] + Var[USDRUB])
    bought_strategy_implied = implied_vol1 ** 2 + implied_vol2 ** 2
    # Calculate realized covariance-adjusted vol of the replication asset (e.g., Var[EURUSD] + 2 * Cov[EURRUB, USDRUB])
    replication_realized = replication_vol ** 2 + 2 * covariance_predict

    # If replication vol is higher => need to buy vol, implied in strategy, vs sold replication
    # (e.g., MOEX vol vs EURUSD offshore realized)
    if replication_realized >= bought_strategy_implied:
        return 'B'
    else:
        return 'S'
