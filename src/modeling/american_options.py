"""
The file provides blueprints to create an american option object
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 5% from notional, the ClassObject.price() will return 0.05)
"""
import numpy as np
import monte_carlo
from european_options import EuropeanCall


class AmericanCall:
    """
    A class of American Call option object.

    ...

    Attributes
    ----------
    term : float
        time_till_maturity in years
    sigma : float
        assumed annualized volatility (constant volatility model)
    spot : float
        current spot price
    initial_spot : float
        spot level at the time of entering into derivative (if entered now, set equal to current_spot)
    rf : float
        available borrowing/lending rate (set to difference in rates for FX options)
    strike_price : float
        option's strike, calcultaed from moneyness in decimal from initial spot (strike_price / initial_spot)

    Methods
    -------
    option_premium():
        Returns option premium in currency via MonteCarlo simulation, comparing at each step PnL of holding option
        further (EuropeanCall price = intrinsic value + time value) vs PnL of executing option now (intrinsic value).

        Note: Should produce same value as EuropeanCall.price() for an asset without interim payments.

    delta():
        Returns option delta by calculating numerically change in option value (dV) at given spot change (dS).

    price():
        Returns option price in % from initial_spot.

        Note: Should produce same value as EuropeanCall.price() for an asset without interim payments.

    bid(spread_from_mid_price):
        Returns price, at which not axed buyer will purchase this option (given specified commission).

    offer(spread_from_mid_price):
        Returns price, at which not axed seller will sell this option (given specified commission).
    """
    def __init__(self, **kwargs):
        self.term = kwargs['time_till_maturity']
        self.sigma = kwargs['underlying_volatility']
        self.spot = kwargs['current_spot']
        self.initial_spot = kwargs['initial_spot_fixing']
        self.rf = kwargs['risk_free_rate']
        self.strike_price = kwargs['strike_decimal'] * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Monte Carlo simulation
    def option_premium(self, spot_change: float = 0):
        """
        Premium of the option (in price space) via MonteCarlo simulation, comparing at each step PnL of holding option
        further (EuropeanCall price = intrinsic value + time value) vs PnL of executing option now (intrinsic value).

        Returns
        -------
        call_price : float
            Option premium.
        """
        spot_used = self.spot + spot_change
        simulated = monte_carlo.geometric_brownian_motion(spot=spot_used, years=self.term, mean=self.rf, vol=self.sigma)
        option_value = []
        for path in simulated:
            for i in range(len(path)):
                spot = path[i]
                exercise_pnl = max(spot - self.strike_price, 0)
                hold_pnl = EuropeanCall(current_spot=spot, initial_spot_fixing=self.initial_spot,
                                        risk_free_rate=self.rf, strike_decimal=self.strike_price / self.initial_spot,
                                        time_till_maturity=(self.term - i / 252),
                                        underlying_volatility=self.sigma).option_premium()
                if i < len(path) - 1:
                    if exercise_pnl > hold_pnl:
                        option_value.append(exercise_pnl)
                        break
                else:
                    option_value.append(exercise_pnl)

        call_price = np.mean(option_value)
        return call_price

    def delta(self, spot_change: float = 0.01):
        """
        Option delta (dV/dS) via calculating numerically change in option value (dV) at given spot change (dS)

        Returns
        -------
        delta : float
            Option delta.
        """
        opt_value_basic = self.option_premium()
        opt_value_changed = self.option_premium(spot_change=spot_change)
        return (opt_value_changed - opt_value_basic) / spot_change

    # Option price as decimal fraction of the initial spot
    def price(self):
        """
        Option price in % from initial_spot_fixing. Premium is calculated by option_premium method.

        Returns
        -------
        price : float
            Option price in %.
        """
        return self.option_premium() / self.initial_spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        """
        Bid to purchase this option.

        Returns
        -------
        bid : float
            Bid for option in %.
        """
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        """
        Offer to sell this option.

        Returns
        -------
        offer : float
            Offer for option in %.
        """
        return self.price() + spread_from_mid_price


class AmericanPut:
    """
    A class of American Put option object.

    ...

    Attributes
    ----------
    term : float
        time_till_maturity in years
    sigma : float
        assumed annualized volatility (constant volatility model)
    spot : float
        current spot price
    initial_spot : float
        spot level at the time of entering into derivative (if entered now, set equal to current_spot)
    rf : float
        available borrowing/lending rate (set to difference in rates for FX options)
    strike_price : float
        option's strike, calcultaed from moneyness in decimal from initial spot (strike_price / initial_spot)

    Methods
    -------
    option_premium():
        Returns option premium in currency via MonteCarlo simulation, comparing at each step PnL of holding option
        further (EuropeanPut price = intrinsic value + time value) vs PnL of executing option now (intrinsic value).

    delta():
        Returns option delta by calculating numerically change in option value (dV) at given spot change (dS).

    price():
        Returns option price in % from initial_spot.

    bid(spread_from_mid_price):
        Returns price, at which not axed buyer will purchase this option (given specified commission).

    offer(spread_from_mid_price):
        Returns price, at which not axed seller will sell this option (given specified commission).
    """
    def __init__(self, **kwargs):
        self.term = kwargs['time_till_maturity']
        self.sigma = kwargs['underlying_volatility']
        self.spot = kwargs['current_spot']
        self.initial_spot = kwargs['initial_spot_fixing']
        self.rf = kwargs['risk_free_rate']
        self.strike_price = kwargs['strike_decimal'] * self.initial_spot

    # Function calculates premium of the option (in price space) by simple Monte Carlo simulation
    def option_premium(self, spot_change: float = 0):
        """
        Premium of the option (in price space) via MonteCarlo simulation, comparing at each step PnL of holding option
        further (EuropeanCall price = intrinsic value + time value) vs PnL of executing option now (intrinsic value).

        Returns
        -------
        put_price : float
            Option premium.
        """
        spot_used = self.spot + spot_change
        simulated = monte_carlo.geometric_brownian_motion(spot=spot_used, years=self.term, mean=0, vol=self.sigma)
        option_value = []
        for path in simulated:
            for i in range(len(path)):
                spot = path[i]
                exercise_pnl = max(self.strike_price - spot, 0)
                hold_pnl = EuropeanCall(current_spot=spot, initial_spot_fixing=self.initial_spot,
                                        risk_free_rate=self.rf, strike_decimal=self.strike_price / self.initial_spot,
                                        time_till_maturity=(self.term - i / 252),
                                        underlying_volatility=self.sigma).option_premium()
                if i < len(path) - 1:
                    if exercise_pnl >= hold_pnl:
                        option_value.append(exercise_pnl)
                        break
                else:
                    option_value.append(exercise_pnl)

        put_price = np.mean(option_value)
        return put_price

    def delta(self, basic_spot_change_decimal: float = 0.01):
        """
        Option delta (dV/dS) via calculating numerically change in option value (dV) at given spot change (dS)

        Returns
        -------
        delta : float
            Option delta.
        """
        opt_value_basic = self.option_premium()
        opt_value_changed = self.option_premium(spot_change=basic_spot_change_decimal)
        return (opt_value_changed - opt_value_basic) / basic_spot_change_decimal

    # Option price as decimal fraction of the initial spot
    def price(self):
        """
        Option price in % from initial_spot_fixing. Premium is calculated by option_premium method.

        Returns
        -------
        price : float
            Option price in %.
        """
        return self.option_premium() / self.initial_spot

    # Get bid as decimal fraction price minus the commission
    def bid(self, spread_from_mid_price):
        """
        Bid to purchase this option.

        Returns
        -------
        bid : float
            Bid for option in %.
        """
        return self.price() - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price):
        """
        Offer to sell this option.

        Returns
        -------
        offer : float
            Offer for option in %.
        """
        return self.price() + spread_from_mid_price
