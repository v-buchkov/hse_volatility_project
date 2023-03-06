from european_options import EuropeanOption, EuropeanCall, EuropeanPut


class Strangle(EuropeanOption):
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

        self.call_strike = kwargs['call_strike']
        self.put_strike = kwargs['put_strike']

        self.european_call = EuropeanCall(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                          strike=self.call_strike, time_till_maturity=self.term, volatility=self.sigma)
        self.european_put = EuropeanPut(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                        strike=self.put_strike, time_till_maturity=self.term, volatility=self.sigma)

    def option_premium(self) -> float:
        """
        Premium of the option (in price space).

        Returns
        -------
        call_price : float
            Option premium.
        """
        return self.european_call.option_premium() + self.european_put.option_premium()

    @property
    def delta(self) -> float:
        """
        Option delta [dV/dS].

        Returns
        -------
        delta : float
            Option delta.
        """
        return self.european_call.delta + self.european_put.delta

    @property
    def gamma(self) -> float:
        """
        Option gamma [d2V/d(S2)].

        Returns
        -------
        gamma : float
            Option gamma.
        """
        return self.european_call.gamma + self.european_put.gamma

    @property
    def vega(self) -> float:
        """
        Option vega [dV/d(sigma)].

        Returns
        -------
        vega : float
            Option vega.
        """
        return self.european_call.vega + self.european_put.vega

    @property
    def theta(self) -> float:
        """
        Option theta [dV/d(term)].

        Returns
        -------
        theta : float
            Option theta.
        """
        return self.european_call.theta + self.european_put.theta

    @property
    def rho(self) -> float:
        """
        Option rho [dV/d(rate)].

        Returns
        -------
        rho : float
            Option rho.
        """
        return self.european_call.rho + self.european_put.rho

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return self.european_call.execute() + self.european_put.execute()


class Straddle(Strangle):
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
        kwargs['call_strike'], kwargs['put_strike'] = 0, 0

        super().__init__(**kwargs)


class CallSpread(EuropeanOption):
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

        self.cap = kwargs['cap']

        self.bought_call = EuropeanCall(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                        strike=self.strike, time_till_maturity=self.term, volatility=self.sigma)
        self.sold_call = EuropeanCall(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                      strike=self.cap, time_till_maturity=self.term, volatility=self.sigma)

    def option_premium(self) -> float:
        """
        Premium of the option (in price space).

        Returns
        -------
        call_price : float
            Option premium.
        """
        return self.bought_call.option_premium() - self.sold_call.option_premium()

    @property
    def delta(self) -> float:
        """
        Option delta [dV/dS].

        Returns
        -------
        delta : float
            Option delta.
        """
        return self.bought_call.delta - self.sold_call.delta

    @property
    def gamma(self) -> float:
        """
        Option gamma [d2V/d(S2)].

        Returns
        -------
        gamma : float
            Option gamma.
        """
        return self.bought_call.gamma - self.sold_call.gamma

    @property
    def vega(self) -> float:
        """
        Option vega [dV/d(sigma)].

        Returns
        -------
        vega : float
            Option vega.
        """
        return self.bought_call.vega - self.sold_call.vega

    @property
    def theta(self) -> float:
        """
        Option theta [dV/d(term)].

        Returns
        -------
        theta : float
            Option theta.
        """
        return self.bought_call.theta - self.sold_call.theta

    @property
    def rho(self) -> float:
        """
        Option rho [dV/d(rate)].

        Returns
        -------
        rho : float
            Option rho.
        """
        return self.bought_call.rho - self.sold_call.rho

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return self.bought_call.execute() - self.sold_call.execute()


class PutSpread(EuropeanOption):
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

        self.floor = kwargs['floor']

        self.bought_put = EuropeanPut(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                      strike=self.strike, time_till_maturity=self.term, volatility=self.sigma)
        self.sold_put = EuropeanPut(spot=self.spot, initial_spot=self.initial_spot, risk_free_rate=self.rf,
                                    strike=self.floor, time_till_maturity=self.term, volatility=self.sigma)

    def option_premium(self) -> float:
        """
        Premium of the option (in price space).

        Returns
        -------
        put_price : float
            Option premium.
        """
        return self.bought_put.option_premium() - self.sold_put.option_premium()

    @property
    def delta(self) -> float:
        """
        Option delta [dV/dS].

        Returns
        -------
        delta : float
            Option delta.
        """
        return self.bought_put.delta - self.sold_put.delta

    @property
    def gamma(self) -> float:
        """
        Option gamma [d2V/d(S2)].

        Returns
        -------
        gamma : float
            Option gamma.
        """
        return self.bought_put.gamma - self.sold_put.gamma

    @property
    def vega(self) -> float:
        """
        Option vega [dV/d(sigma)].

        Returns
        -------
        vega : float
            Option vega.
        """
        return self.bought_put.vega - self.sold_put.vega

    @property
    def theta(self) -> float:
        """
        Option theta [dV/d(term)].

        Returns
        -------
        theta : float
            Option theta.
        """
        return self.bought_put.theta - self.sold_put.theta

    @property
    def rho(self) -> float:
        """
        Option rho [dV/d(rate)].

        Returns
        -------
        rho : float
            Option rho.
        """
        return self.bought_put.rho - self.sold_put.rho

    # Execute the option => get the realization of the payoff function (in price space)
    def execute(self):
        """
        Payoff of the option in currency.

        Returns
        -------
        payoff : float
            Payoff in currency.
        """
        return self.bought_put.execute() - self.sold_put.execute()
