"""Main module for backtesting statistical arbitrage strategy of realized local vol vs realized offshore vol"""
import numpy as np
import datetime as dt
from typing import List, Tuple, Dict
from static_data import PATH, DELTA_SECONDS
from stat_arb.src.working_with_files.preprocessing import get_asset_returns, get_asset_prices
from stat_arb.src.modeling.vol_cov import calculate_vol_realized
from stat_arb.src.modeling.european_options import EuropeanCall
from stat_arb.src.plt.graphs import plot_barchart, plot_line_chart


class BacktesterOffshoreLocalRealized:
    """
    Class creates Backtester object.
    This Backtester is used for testing several hypotheses of local vs offshore volatility (without correlation trade).

    ...

    Attributes
    ----------
    asset : str
        name of the asset that is being tested (e.g., 'EURUSD')
    start : datetime.datetime
        start of the backtesting period
    end : datetime.datetime
        end of the backtesting period
    rf_base_ccy : float
        risk-free rate of the base currency (EUR for EURUSD)
    rf_second_ccy : float
        risk-free rate of the second currency (USD for EURUSD)
    onshore_spread : float
        spread-paid for local standard size deal (e.g., 1 mio EURUSD on MOEX)
    offshore_spread : float
        spread-paid for offshore standard size deal (e.g., 1 mio EURUSD with KZ counterpart)

    Methods
    -------
    get_backtesting_dataset(prices, start, end):
        Returns (timestamp, price) data for the specified time interval
    get_onshore_prices(asset):
        Returns dict of local market historical prices for specified asset, using self.local_price_source
    get_offshore_prices(asset):
        Returns dict of foreign market historical prices for specified asset, using self.foreign_price_source
    standardize_price_data(input_prices):
        Sets all unnecessary time interval values to zero.
        For instance, if we use hourly data, we should set minutes, seconds and microseconds to zero
    get_realized_vol(asset, source, date_start, date_end):
        Returns float of realized vol as standard deviation over the period between date_start and date_end
    delta_hedge_process
    backtest(days_strategy):
        Backtests the strategy of trading lower realized vol versus higher realized vol.
        Realized vols over last n days are compared and the option is assumed to be bought for market with lower one.
        Option is replicated by dynamic delta-hedge with spread_paid and risk-free rate costs
    """
    # Stat Arb trade notional is specified as the class object attribute for now
    # Further we can use notional for modeling spread_paid more accurately
    # (e.g., initial delta hedged spread paid >> further small delta changes spread_paid)
    notional = 1000000
    local_price_source = 'moex'
    foreign_price_source = 'rbi'
    delta_seconds = DELTA_SECONDS

    def __init__(self, asset: str, datetime_start: dt.datetime, datetime_end: dt.datetime, rf_base_ccy: float,
                 rf_second_ccy: float, onshore_spread: float, offshore_spread: float):
        """
        Initializing backtest parameters.

        Parameters
        ----------
            asset : str
                name of the asset that is being tested (e.g., 'EURUSD')
            datetime_start : datetime.datetime
                start of the backtesting period
            datetime_end : datetime.datetime
                end of the backtesting period
            rf_base_ccy : float
                risk-free rate of the base currency (EUR for EURUSD)
            rf_second_ccy : float
                risk-free rate of the second currency (USD for EURUSD)
            onshore_spread : float
                spread-paid for local standard size deal (e.g., 1 mio EURUSD on MOEX)
            offshore_spread : float
                spread-paid for offshore standard size deal (e.g., 1 mio EURUSD with KZ counterpart)

        Returns
        -------
        None
        """
        self.asset = asset

        self.start = datetime_start
        self.end = datetime_end

        self.rf_base_ccy = rf_base_ccy
        self.rf_second_ccy = rf_second_ccy

        self.onshore_spread = onshore_spread
        self.offshore_spread = offshore_spread

        self.hist_vols = []
        self.backtesting_date = []

    @staticmethod
    def get_backtesting_dataset(prices: List[Tuple[dt.datetime, float]], start: dt.datetime,
                                end: dt.datetime) -> List[Tuple[dt.datetime, float]]:
        """
        Returns (timestamp, price) data for the specified time interval.

        Parameters
        ----------
            prices : list
                Initial data.
            start : datetime.datetime
                Starting date.
            end : datetime.datetime
                Ending date.

        Returns
        -------
        dataset_backtest : list
            List of (timestamp, price) tuples for the chosen backtest interval.
        """
        dataset_backtest = []
        for point in prices:
            date, price = point
            if date >= start:
                if date > end:
                    break
                else:
                    dataset_backtest.append((date, price))
            else:
                pass

        return dataset_backtest

    def get_onshore_prices(self) -> Dict[dt.datetime, float]:
        """
        Onshore prices of the asset.

        Returns
        -------
        asset_prices : dict
            Onshore prices {timestamp: price} dict for the asset that is being tested.
        """
        return get_asset_prices(path=PATH, asset=self.asset, price_source=self.local_price_source,
                                delta_seconds=self.delta_seconds)

    def get_offshore_prices(self) -> Dict[dt.datetime, float]:
        """
        Offshore prices of the asset.

        Returns
        -------
        asset_prices : dict
            Offshore prices {timestamp: price} dict for the asset that is being tested.
        """
        return get_asset_prices(path=PATH, asset=self.asset, price_source=self.foreign_price_source,
                                delta_seconds=self.delta_seconds)

    def standardize_price_data(self, input_prices: Dict[dt.datetime, float]) -> List[Tuple[dt.datetime, float]]:
        """
        Sets all unnecessary time interval values to zero.
        For instance, if we use hourly data, we should set minutes, seconds and microseconds to zero.

        Parameters
        ----------
        input_prices : dict
            Prices {timestamp: price} dict with actual timestamps.

        Returns
        -------
        asset_prices : dict
            Prices {timestamp: price} dict with processed timestamps.
        """
        output_prices = []
        for timestamp, price in input_prices.items():

            if self.delta_seconds < 60:
                timestamp = timestamp.replace(microsecond=0)
            elif self.delta_seconds < 60 * 60:
                timestamp = timestamp.replace(second=0, microsecond=0)
            elif self.delta_seconds < 12 * 60 * 60:
                timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
            elif self.delta_seconds < 24 * 60 * 60:
                timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                timestamp = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            output_prices.append((timestamp, price))

        return output_prices

    def get_realized_vol(self, source: str, date_start: dt.datetime, date_end: dt.datetime) -> float:
        """
        Realized vol as standard deviation over the period between date_start and date_end.

        Parameters
        ----------
            source : str
                asset price source
            date_start : datetime.datetime
                first date of the period window
            date_end : datetime.datetime
                last date of the period window

        Returns
        -------
        realized_vol : float
            Realized volatility of the asset that is being tested over specified period.
        """
        asset_returns = get_asset_returns(path=PATH, asset=self.asset, price_source=source,
                                          delta_seconds=self.delta_seconds)
        # Get returns that lie inside the specified time interval
        returns_for_vol = [r for date, r in asset_returns.items() if date_start <= date <= date_end]

        return calculate_vol_realized(returns_for_vol, delta_seconds=self.delta_seconds)

    def delta_hedge_process(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                            backtest_data: List[Tuple[dt.datetime, float]]) -> Tuple[float, float]:
        """
        Provides simulation of the delta hedge process on given prices for an option, specified by the backtest params.
        Returns option cost and transaction costs paid (separately).

        Parameters
        ----------
            price_source : str
                Source for prices, for which the option is being replicated.
            spot_start : float
                a
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list


        Returns
        -------
        opt_cost, transaction_cost : tuple
            Option cost for the specified option and transaction costs paid (separately)
        """
        # First delta of the position is exactly zero => first trade would be on the full delta of the option
        delta_old = 0

        # Initialize costs (zero before the first trade)
        cost = 0
        rf_paid = 0
        spread_paid = 0

        # Will charge rf rate for daily residual of delta only (intraday trades are not used for funding cost)
        daily_rf_difference = (self.rf_second_ccy - self.rf_base_ccy) / 365
        rf_accumulation_start = start

        # Now iterate over available points for this backtesting period => will generate PnL[t]
        for point in backtest_data[:-1]:
            curr_date, spot = point

            # Get dates of the realized vol for delta calculation (for last n days, where n = days_strategy)
            vol_date_start = curr_date - (end - start)
            vol_date_end = curr_date

            # Calculate realized vols over last days_strategy
            vol_realized = self.get_realized_vol(source=price_source, date_start=vol_date_start, date_end=vol_date_end)

            # Check that vol point is available
            if (not np.isnan(vol_realized)) and (vol_realized != 0):
                # Calculate time until maturity
                till_maturity = (end - curr_date) / dt.timedelta(days=252)

                # Specify option parameters
                # Rf rate = difference (subtract base from second - e.g., subtract CNH rate from RUB rate)
                opt = {'time_till_maturity': till_maturity, 'current_spot': spot, 'initial_spot_fixing': spot_start,
                       'risk_free_rate': self.rf_second_ccy - self.rf_base_ccy, 'strike_decimal': 1,
                       'underlying_volatility': vol_realized}

                # Initialize options objects from class EuropeanCall
                call_obj = EuropeanCall(**opt)
                # For all days inside period support delta of the portfolio equal to the corresponding option
                delta = call_obj.delta()

                if not np.isnan(delta - delta_old):
                    # Get spot cost for the delta_hedge
                    delta_hedge_cost = self.notional * (delta - delta_old) * spot
                    # Get spread paid for hedging the delta difference
                    spread = abs(self.notional * (delta - delta_old) * self.onshore_spread)

                    # Enter the trade to hedge delta, only if delta difference is large enough =>
                    # => need E(loss unhedged position) >= spread => need E(gamma/2 * (dS)**2) >= spread =>
                    # => as sigma ** 2 = E(r**2) - E(r)**2, we have (gamma/2 * sigma **2) >= spread
                    # Sigma should be taken for the specified time interval => use delta seconds
                    variance_delta_time = vol_realized ** 2 / (252 * 9 * 60 * 60 / self.delta_seconds)
                    if self.notional * call_obj.gamma() / 2 * variance_delta_time >= spread:
                        # Dynamically calculate delta-hedge cost = PnL on the option
                        # Same as calculating value of the portfolio = [delta * spot + risk-free asset]
                        cost += delta_hedge_cost
                        # Add spread-paid (specified as % of notional traded)
                        spread_paid += spread_paid

                        # Charge daily rf rate, only if timedelta is larger than 1 day
                        if curr_date - rf_accumulation_start >= dt.timedelta(days=1):
                            # Add daily risk-free rate paid / received (without compounding, as rf == funding)
                            rf_paid += self.notional * delta * daily_rf_difference
                            rf_accumulation_start = curr_date

                        # Reinitialize delta (as already hedged)
                        delta_old = delta

        # At the last date we need to calculate PnL => unwind FX Spot position in full and get PnL
        # In general it is exactly the same as just calculate PnL as [delta * (spot[t] / spot[0] - 1)]
        cost += -delta_old * self.notional * backtest_data[-1][1]
        spread_paid += -delta_old * abs(self.notional * self.onshore_spread)

        # Option cost = PnL on spot dynamic delta replication + risk-free rate paid
        opt_cost = cost + rf_paid
        transaction_cost = spread_paid

        return opt_cost, transaction_cost

    def backtest(self, days_strategy: int) -> List[float]:
        """
        Backtests the strategy of trading lower realized vol versus higher realized vol.
        Realized vols over last n days are compared and the option is assumed to be bought for market with lower one.
        Option is replicated by dynamic delta-hedge with spread_paid and risk-free rate costs.

        Parameters
        ----------
            days_strategy : int
                days for backtesting (time till maturity of the bought option)

        Returns
        -------
        strategy_pnl : float
            PnL for the basktested strategy.
        """
        days_backtesting_period = (self.end - self.start).days
        onshore_prices = self.get_onshore_prices()
        offshore_prices = self.get_offshore_prices()

        # For hourly data set minutes, seconds and microseconds to zero (returns tuple)
        onshore_prices = self.standardize_price_data(onshore_prices)
        offshore_prices = self.standardize_price_data(offshore_prices)

        # Get sorted by date list of tuples
        onshore_prices = sorted(onshore_prices, key=lambda x: x[0])
        offshore_prices = sorted(offshore_prices, key=lambda x: x[0])

        # Get unique set of dates, where at least one datapoint is present
        onshore_available_days = sorted(set([key.date() for key in [x[0] for x in onshore_prices]]))
        offshore_available_days = sorted(set([key.date() for key in [x[0] for x in offshore_prices]]))

        trades_pnl = []

        # Iterate over all available days - change the date of start for backtesting (no clustering of PnL)
        for t in range(1, days_backtesting_period - days_strategy):
            # Get date of testing as first available date + t days, iterate over t
            date_start = self.start + dt.timedelta(days=t)
            date_end = date_start + dt.timedelta(days=days_strategy)

            # Check if both assets were trading on this day (if not => just do not backtest from this day)
            if (date_start.date() in onshore_available_days) and (date_start.date() in offshore_available_days):
                # Get datasets for testing
                onshore_dataset_backtest = self.get_backtesting_dataset(onshore_prices, date_start, date_end)
                offshore_dataset_backtest = self.get_backtesting_dataset(offshore_prices, date_start, date_end)

                # Get starting spots (day of starting the backtest)
                spot_onshore_start = onshore_dataset_backtest[0][1]
                spot_offshore_start = offshore_dataset_backtest[0][1]

                # Get realized vol of days_strategy before entering the backtest
                vol_onshore = self.get_realized_vol(source=self.local_price_source,
                                                    date_start=date_start - dt.timedelta(days=days_strategy+1),
                                                    date_end=date_start)
                vol_offshore = self.get_realized_vol(source=self.foreign_price_source,
                                                     date_start=date_start - dt.timedelta(days=days_strategy+1),
                                                     date_end=date_start)

                # Get results of delta-hedging onshore and offshore option replication
                opt_onshore, trans_cost_onshore = self.delta_hedge_process(price_source=self.local_price_source,
                                                                           start=date_start, end=date_end,
                                                                           spot_start=spot_onshore_start,
                                                                           backtest_data=onshore_dataset_backtest)
                opt_offshore, trans_cost_offshore = self.delta_hedge_process(price_source=self.foreign_price_source,
                                                                             start=date_start, end=date_end,
                                                                             spot_start=spot_offshore_start,
                                                                             backtest_data=offshore_dataset_backtest)

                total_transaction_cost = trans_cost_onshore + trans_cost_offshore

                self.hist_vols.append((vol_onshore, vol_offshore))

                if vol_offshore > vol_onshore:
                    # PnL is cost of more expensive option minus cost of less expensive one
                    # Therefore, need to subtract cost of lower vol market from the cost of higher vol one
                    # Spread-paid is a commission => just subtract, as paid on both sides symmetrically
                    pnl = opt_offshore - opt_onshore - total_transaction_cost
                else:
                    pnl = opt_onshore - opt_offshore - total_transaction_cost

                # Append to the list of trade dates (for graphs)
                self.backtesting_date.append(date_start)
                trades_pnl.append(pnl)

        return trades_pnl
