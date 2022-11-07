"""
The file provides blueprints to create an european option object
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 5% from notional, the ClassObject.price() will return 0.05)
"""
import scipy.stats
import numpy as np
from abc import abstractmethod


class VanillaOptions:
    def __init__(self, **kwargs):
        self.term = kwargs['time_till_maturity']
        self.sigma = kwargs['underlying_volatility']
        self.spot = kwargs['current_spot']
        self.initial_spot = kwargs['initial_spot_fixing']
        self.rf = kwargs['risk_free_rate']
        self.strike_price = kwargs['strike_decimal'] * self.initial_spot

    @abstractmethod
    @property
    def option_premium(self):
        return 0

    @abstractmethod
    def execute(self):
        return 0

    def calculate_call_delta(self):
        """
        Option delta (dV/dS) via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        delta : float
            Option delta.
        """
        d1 = (np.log(self.spot / self.strike_price) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))

        cdf_d1 = scipy.stats.norm.cdf(d1)

        return cdf_d1

    def calculate_call_gamma(self):
        """
        Option gamma (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        gamma : float
            Option gamma.
        """
        d1 = (np.log(self.spot / self.strike_price) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))

        pdf_d1 = scipy.stats.norm.pdf(d1)

        return pdf_d1 / (self.spot * self.sigma * self.term)

    def calculate_call_premium(self):
        """
        Premium of the option (in price space) by simple Black-Scholes-Merton.

        Returns
        -------
        call_price : float
            Option premium.
        """
        d1 = (np.log(self.spot / self.strike_price) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)

        cdf_d1 = scipy.stats.norm.cdf(d1)
        cdf_d2 = scipy.stats.norm.cdf(d2)

        call_price = self.spot * cdf_d1 - cdf_d2 * self.strike_price * np.exp(-self.rf * self.term)

        return call_price

    @property
    # Option price as decimal fraction of the initial spot
    def price(self):
        """
        Option price in % from initial_spot_fixing. Premium is calculated by option_premium method.

        Returns
        -------
        price : float
            Option price in %.
        """
        return self.option_premium / self.initial_spot

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
        Future Value of PnL of the trade as (payoff - premium - commission) in % of initial_spot_fixing.

        Returns
        -------
        pnl : float
            PnL in %.
        """
        option_pnl = self.execute() / self.initial_spot
        future_value_premium = (self.option_premium / self.spot - commission_paid) * (1 + self.rf) ** self.term

        return option_pnl - future_value_premium


class EuropeanCall(VanillaOptions):
    """
    A class of European Call option object.

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
        Returns option premium in currency via analytical form solution of Black-Scholes-Merton.

    delta():
        Returns option delta (dV/dS) via analytical form solution of Black-Scholes-Merton.

    gamma():
        Returns option gamma (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    price():
        Returns option price in % from initial_spot.

    bid(spread_from_mid_price):
        Returns price, at which not axed buyer will purchase this option (given specified commission).

    offer(spread_from_mid_price):
        Returns price, at which not axed seller will sell this option (given specified commission).

    execute():
        Returns payoff of the option in currency.

    final_result(commission_paid=0):
        Returns PnL of the trade as (payoff - premium - commission) in % of initial_spot.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def option_premium(self):
        return self.calculate_call_premium()

    @property
    def delta(self):
        return self.calculate_call_delta()

    @property
    def gamma(self):
        return self.calculate_call_gamma()

    @property
    def vega(self):
        return self.calculate_call_vega()

    @property
    def theta(self):
        return self.calculate_call_theta()

    @property
    def rho(self):
        return self.calculate_call_rho()

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if final_fixing > initial_fixing:
            return final_fixing - initial_fixing
        else:
            return 0


class EuropeanPut(VanillaOptions):
    """
    A class of European Put option object.

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
        Returns option premium in currency via analytical form solution of Black-Scholes-Merton.

    delta():
        Returns option delta (dV/dS) via analytical form solution of Black-Scholes-Merton.

    gamma():
        Returns option gamma (d2V/dS2) via analytical form solution of Black-Scholes-Merton.

    price():
        Returns option price in % from initial_spot.

    bid(spread_from_mid_price):
        Returns price, at which not axed buyer will purchase this option (given specified commission).

    offer(spread_from_mid_price):
        Returns price, at which not axed seller will sell this option (given specified commission).

    execute():
        Returns payoff of the option in currency.

    final_result(commission_paid=0):
        Returns PnL of the trade as (payoff - premium - commission) in % of initial_spot.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def option_premium(self):
        return self.calculate_call_premium() - self.spot + self.strike_price * np.exp(-self.rf * self.term)

    @property
    def delta(self):
        return self.calculate_call_delta() - 1

    @property
    def gamma(self):
        return self.calculate_call_gamma()

    @property
    def vega(self):
        return self.calculate_call_vega()

    @property
    def theta(self):
        return self.calculate_call_theta()

    @property
    def rho(self):
        return self.calculate_call_rho()

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        initial_fixing = self.initial_spot
        final_fixing = self.spot

        if final_fixing < initial_fixing:
            return initial_fixing - final_fixing
        else:
            return 0


if __name__ == '__main__':
    opt_params = {'time_till_maturity': 0.25, 'current_spot': 63.50, 'initial_spot_fixing': 63.50,
                  'risk_free_rate': 0.075 - 0.024, 'strike_decimal': 1,
                  'underlying_volatility': 0.3281}
    opt = EuropeanCall(**opt_params)

    print(opt.price)
