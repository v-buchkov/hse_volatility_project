"""
The file provides blueprints to create an european option object
----------------------------
The convention for all fractions is decimals
(e.g., if the price is 5% from notional, the ClassObject.price() will return 0.05)
"""
import scipy.stats
import numpy as np
from abc import abstractmethod


class EuropeanOption:
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
    def option_premium(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No option premium specified in the child class')

    @property
    @abstractmethod
    def delta(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No delta formula specified in the child class')

    @property
    @abstractmethod
    def gamma(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No gamma formula specified in the child class')

    @property
    @abstractmethod
    def vega(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No vega formula specified in the child class')

    @property
    @abstractmethod
    def theta(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No theta formula specified in the child class')

    @property
    @abstractmethod
    def rho(self) -> float:
        """
        Formula for calculating premium of the option. Might be an analytical solution or an algorithm.

        Required to create the option object. Returns AssertionError, if child class doesn't have this property.
        """
        raise AssertionError('No rho formula specified in the child class')

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

    @property
    def _call_premium(self):
        """
        Premium of the call option (in price space) by simple Black-Scholes-Merton.

        Returns
        -------
        call_price : float
            Option premium.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)

        cdf_d1 = scipy.stats.norm.cdf(d1)
        cdf_d2 = scipy.stats.norm.cdf(d2)

        call_price = self.spot * cdf_d1 - cdf_d2 * self.strike * np.exp(-self.rf * self.term)

        return call_price

    @property
    def _call_delta(self):
        """
        Call option delta [dV/dS] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        delta : float
            Option delta.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))

        cdf_d1 = scipy.stats.norm.cdf(d1)

        return cdf_d1

    @property
    def _call_gamma(self):
        """
        Call option gamma [d2V/d(S2)] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        gamma : float
            Option gamma.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))

        pdf_d1 = scipy.stats.norm.pdf(d1)

        return pdf_d1 / (self.spot * self.sigma * self.term)

    @property
    def _call_vega(self):
        """
        Call option vega [dV/d(sigma)] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        vega : float
            Option vega.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))

        pdf_d1 = scipy.stats.norm.pdf(d1)

        return self.spot * np.sqrt(self.term) * pdf_d1

    @property
    def _call_theta(self):
        """
        Call option theta [dV/d(term)] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        theta : float
            Option theta.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)

        cdf_d2 = scipy.stats.norm.cdf(d2)

        pdf_d1 = scipy.stats.norm.pdf(d1)

        return -(self.spot * pdf_d1 * self.sigma) / (2 * np.sqrt(self.term)) - \
               self.rf * self.strike * np.exp(-self.rf * self.term) * cdf_d2

    @property
    def _call_rho(self):
        """
        Call option rho [dV/d(rate)] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        rho : float
            Option rho.
        """
        d1 = (np.log(self.spot / self.strike) + (self.rf + self.sigma ** 2 / 2) * self.term) / \
             (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)

        cdf_d2 = scipy.stats.norm.cdf(d2)

        return self.strike * self.term * np.exp(-self.rf * self.term) * cdf_d2


class EuropeanCall(EuropeanOption):
    """
    A class of European Call option object. Inherited from EuropeanOption class.

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

    def option_premium(self) -> float:
        """
        Premium of the option (in price space).

        Returns
        -------
        call_price : float
            Option premium.
        """
        return self._call_premium

    @property
    def delta(self) -> float:
        """
        Option delta [dV/dS].

        Returns
        -------
        delta : float
            Option delta.
        """
        return self._call_delta

    @property
    def gamma(self) -> float:
        """
        Option gamma [d2V/d(S2)].

        Returns
        -------
        gamma : float
            Option gamma.
        """
        return self._call_gamma

    @property
    def vega(self) -> float:
        """
        Option vega [dV/d(sigma)].

        Returns
        -------
        vega : float
            Option vega.
        """
        return self._call_vega

    @property
    def theta(self) -> float:
        """
        Option theta [dV/d(term)].

        Returns
        -------
        theta : float
            Option theta.
        """
        return self._call_theta

    @property
    def rho(self) -> float:
        """
        Option rho [dV/d(rate)].

        Returns
        -------
        rho : float
            Option rho.
        """
        return self._call_rho

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return max(self.spot - self.strike, 0)


class EuropeanPut(EuropeanOption):
    """
    A class of European Put option object. Inherited from EuropeanOption class.

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

    def option_premium(self) -> float:
        """
        Premium of the option (in price space).

        Returns
        -------
        call_price : float
            Option premium.
        """
        return self._call_premium - self.spot + self.strike * np.exp(-self.rf * self.term)

    @property
    def delta(self) -> float:
        """
        Option delta [dV/dS].

        Returns
        -------
        delta : float
            Option delta.
        """
        return self._call_delta - 1

    @property
    def gamma(self) -> float:
        """
        Option gamma [d2V/d(S2)].

        Returns
        -------
        gamma : float
            Option gamma.
        """
        return self._call_gamma

    @property
    def vega(self) -> float:
        """
        Option vega [dV/d(sigma)].

        Returns
        -------
        vega : float
            Option vega.
        """
        return self._call_vega

    @property
    def theta(self) -> float:
        """
        Option theta [dV/d(term)].

        Returns
        -------
        theta : float
            Option theta.
        """
        # Assuming that [dS/dt = 0]
        return self._call_theta - self.strike * self.rf * np.exp(-self.rf * self.term)

    @property
    def rho(self) -> float:
        """
        Option rho [dV/d(rate)].

        Returns
        -------
        rho : float
            Option rho.
        """
        return self._call_rho - self.strike * self.term * np.exp(-self.rf * self.term)

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self) -> float:
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return max(self.strike - self.spot, 0)
