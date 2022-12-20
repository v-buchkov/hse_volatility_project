"""
The file provides blueprints to create an american option object
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 5% from notional, the ClassObject.price() will return 0.05)
"""
import numpy as np
import monte_carlo
from european_options import EuropeanCall, EuropeanPut
from abc import abstractmethod


class AmericanOption:
    """
    A class of Vanilla European Option. Determines basic (shared) methods for creating an European option object.

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
    strike : float
        option's strike, calcultaed from moneyness in decimal from initial spot (strike_price / initial_spot)

    Methods
    -------
    option_premium():
        Returns option premium in currency via analytical form solution of Black-Scholes-Merton.

    delta:
        Returns option delta (dV/dS) via analytical form solution of Black-Scholes-Merton.

    gamma:
        Returns option gamma (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    vega:
        Returns option vega (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    gamma:
        Returns option vega (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    gamma:
        Returns option gamma (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    price:
        Returns option price in % from initial_spot.

    bid(spread_from_mid_price):
        Returns price, at which not axed buyer will purchase this option (given specified commission).

    offer(spread_from_mid_price):
        Returns price, at which not axed seller will sell this option (given specified commission).

    execute:
        Returns payoff of the option in currency.

    final_result(commission_paid=0):
        Returns PnL of the trade as (payoff - premium - commission) in % of initial_spot.
    """
    def __init__(self, **kwargs):

        self.term = kwargs['time_till_maturity']
        self.sigma = kwargs['volatility']
        self.spot = kwargs['spot']
        self.initial_spot = kwargs['initial_spot']
        self.rf = kwargs['risk_free_rate']
        self.strike = kwargs['strike']

        # List of arguments that should be non-negative
        non_negative_args = [self.term, self.sigma, self.spot, self.initial_spot, self.rf, self.strike]

        # Check for non-negativity
        for a in non_negative_args:
            var_name = f'{a=}'.rstrip('=')
            assert a > 0, f'{var_name} should be a positive decimal!'

        # Moenyness level in decimal
        self.moneyness = self.strike / self.spot

    @abstractmethod
    def option_premium(self, spot_change: float = 0) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No option premium specified in the child class')

    @property
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

    @abstractmethod
    def execute(self) -> float:
        """
        Payoff of the option in currency.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No payoff function specified in the child class')

    # Option price as decimal fraction of the initial spot
    @property
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
    def bid(self, spread_from_mid_price: float):
        """
        Bid to purchase this option.

        Returns
        -------
        bid : float
            Bid for option in %.
        """
        return self.price - spread_from_mid_price

    # Get offer as decimal fraction price plus the commission
    def offer(self, spread_from_mid_price: float):
        """
        Offer to sell this option.

        Returns
        -------
        offer : float
            Offer for option in %.
        """
        return self.price + spread_from_mid_price

    # Final result => payoff in decimal fraction, excluding the commissions and initial premium paid
    def final_result(self, commission_paid: float = 0):
        """
        PnL of the trade as (payoff - premium - commission) in % of initial_spot_fixing.

        Returns
        -------
        pnl : float
            PnL in %.
        """
        return self.execute() / self.initial_spot - (self.option_premium() / self.spot -
                                                     commission_paid) * (1 + self.rf) ** self.term

    # Function calculates premium of the option (in price space) by simple Monte Carlo simulation
    def _calculate_premium_by_monte_carlo(self, ComparableOption, spot_change: float = 0):
        """
        Premium of the option (in price space) via MonteCarlo simulation, comparing at each step PnL of holding option
        further (EuropeanCall price = intrinsic value + time value) vs PnL of executing option now (intrinsic value).

        Returns
        -------
        call_price : float
            Option premium.
        """
        # Spot level, given some shift (0 by default). Used for calculating the
        spot_used = self.spot + spot_change
        # Create the simulated paths
        simulated = monte_carlo.geometric_brownian_motion(spot=spot_used, years=self.term, mean=self.rf,
                                                          vol=self.sigma)
        # Option values paths
        option_value = []
        for path in simulated:
            for i in range(len(path)):
                spot = path[i]
                option = ComparableOption(spot=spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                          strike=self.strike, time_till_maturity=(self.term - i / 252),
                                          volatility=self.sigma)
                # Exercise PnL => just payoff function
                exercise_pnl = option.execute()
                # Holding PnL => the value of plain European Call
                hold_pnl = option.option_premium()
                if i < len(path) - 1:
                    # If exercise PnL is higher, then
                    if exercise_pnl > hold_pnl:
                        # If exercise PnL is higher => optimal to exercise => exit from simulation
                        option_value.append(exercise_pnl)
                        break
                    # Else => continue to hold
                else:
                    # Last day => exercise anyway (similar to European Option)
                    option_value.append(exercise_pnl)

        # Price is the average of all scenarios (Monte Carlo provides the unbiased estimator)
        option_price = np.mean(option_value)

        return option_price


class AmericanCall(AmericanOption):
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
        super().__init__(**kwargs)

    def option_premium(self, spot_change: float = 0) -> float:
        return self._calculate_premium_by_monte_carlo(EuropeanCall, spot_change=spot_change)

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return EuropeanCall(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf, strike=self.strike,
                            time_till_maturity=self.term, volatility=self.sigma).execute()


class AmericanPut(AmericanOption):
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
        super().__init__(**kwargs)

    def option_premium(self, spot_change: float = 0) -> float:
        return self._calculate_premium_by_monte_carlo(EuropeanPut, spot_change=spot_change)

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return EuropeanPut(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf, strike=self.strike,
                           time_till_maturity=self.term, volatility=self.sigma).execute()
