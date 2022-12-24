"""Main module for backtesting statistical arbitrage strategy of realized local vol vs realized offshore vol"""
import datetime as dt
from abc import abstractmethod
from typing import List, Tuple, Dict, Union

import scipy.stats
import numpy as np

from static_data import PATH, DELTA_SECONDS

from src.working_with_files.preprocessing import get_asset_returns, get_asset_prices
from src.modeling.vol_cov import calculate_vol_realized
from src.modeling.european_options import EuropeanCall


class Backtester:
    """
    Class creates Backtester object, which carries basic features for every backtest of every strategy.

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
    onshore_price_source : str
        string name of the source for onshore prices
    offshore_price_source : str
        string name of the source for offshore prices

    Methods
    -------
    _get_backtesting_dataset(prices, start, end):
        Returns (timestamp, price) data for the specified time interval
    _standardize_price_data(input_prices):
        Sets all unnecessary time interval values to zero.
        For instance, if we use hourly data, we should set minutes, seconds and microseconds to zero
    _get_realized_vol(asset, source, date_start, date_end):
        Returns float of realized vol as standard deviation over the period between date_start and date_end
    backtest_path:
        Backtested strategy path of PnLs per each period taken (can be a PnL or 0, if we decided not to trade)
    pnl_distribution_by_trades:
        Distribution of PnLs only for the cases, when a trade was conducted (decided to trade)
    pnl_distribution_cumulative:
        Cumulative distribution of PnLs only for the cases, when a trade was conducted (decided to trade)
    pnl_total:
        Sum of all trades' PnLs
    pnl_mean:
        Average of all trades' PnLs
    pnl_std:
        Standard deviation of all trades' PnLs
    pnl_sharpe:
        Sharpe ratio of all trades' PnLs
    t_statistic:
        T-statistic for the test, if the backtest generates PnLs that are consistently larger than zero
    t_test_p_value:
        P-value for the Student t-test, if the backtest generates PnLs that are consistently larger than zero
    t_test_result_significant:
        Result for the Student t-test, if the backtest generates PnLs that are consistently larger than zero
    backtest():
        Backtesting process that generates strategy PnL and saves the time series to self._backtest_pnl array
    trading_strategy():
        Conducts the process of trading strategy simulation for one specific time interval (one iteration of backtest)
    """
    # Stat Arb trade notional is specified as the class object attribute for now
    # Further we can use notional for modeling spread_paid more accurately
    # (e.g., initial delta hedged spread paid >> further small delta changes spread_paid)
    notional = 1000000
    delta_seconds = DELTA_SECONDS

    def __init__(self, asset: str, datetime_start: dt.datetime, datetime_end: dt.datetime, rf_base_ccy: float,
                 rf_second_ccy: float, onshore_spread: float, offshore_spread: float,
                 onshore_price_source: str = 'moex', offshore_price_source: str = 'rbi'):
        """
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

        self.onshore_price_source = onshore_price_source
        self.offshore_price_source = offshore_price_source

        self._backtest_pnl = []

    @staticmethod
    def _get_backtesting_dataset(prices: List[Tuple[dt.datetime, float]], start: dt.datetime,
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

    def _standardize_price_data(self, input_prices: Dict[dt.datetime, float]) -> List[Tuple[dt.datetime, float]]:
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

    def _get_realized_vol(self, source: str, date_start: dt.datetime, date_end: dt.datetime) -> float:
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

    @property
    def backtest_path(self) -> List[float]:
        """
        Backtested strategy path of PnLs per each period taken (can be a PnL or 0, if we decided not to trade).

        Returns
        -------
        path : list
            List of PnLs per each backtested period (a PnL or 0).
        """
        assert len(self._backtest_pnl) > 0, "No trades were backtested yet. Please, use backtest() method first"

        return self._backtest_pnl

    @property
    def pnl_distribution_by_trades(self) -> List[float]:
        """
        Distribution of PnLs only for the cases, when a trade was conducted (decided to trade).

        Returns
        -------
        pnl_distribution : list
            List of PnLs per each trade (a PnL).
        """
        return [p for p in self.backtest_path if p != 0]

    @property
    def pnl_distribution_cumulative(self) -> List[float]:
        """
        Cumulative distribution of PnLs only for the cases, when a trade was conducted (decided to trade).

        Returns
        -------
        pnl_distribution : list
            List of cumulative PnL change per each trade (a PnL).
        """
        return [sum(self.backtest_path[:i]) for i in range(2, len(self.backtest_path))]

    @property
    def pnl_total(self) -> float:
        """
        Sum of all trades' PnLs.

        Returns
        -------
        pnl_total : list
            Sum of all trades' PnLs.
        """
        return sum(self.pnl_distribution_by_trades)

    @property
    def pnl_mean(self) -> float:
        """
        Average of all trades' PnLs.

        Returns
        -------
        pnl_total : list
            Average of all trades' PnLs.
        """
        return float(np.mean(self.pnl_distribution_by_trades))

    @property
    def pnl_std(self) -> float:
        """
        Standard deviation of all trades' PnLs.

        Returns
        -------
        pnl_total : list
            Standard deviation of all trades' PnLs.
        """
        return float(np.std(self.pnl_distribution_by_trades))

    @property
    def pnl_sharpe(self) -> float:
        """
        Sharpe ratio of all trades' PnLs.

        Returns
        -------
        pnl_total : list
            Sharpe ratio of all trades' PnLs.
        """
        return (self.pnl_mean - self.rf_base_ccy) / self.pnl_std

    @property
    def t_statistic(self) -> float:
        """
        T-statistic for the test, if the backtest generates PnLs that are consistently larger than zero.

        Returns
        -------
        t_statistic : float
            T-statistic for the test (without rounding).
        """
        return self.pnl_mean / self.pnl_std * np.sqrt(len(self.pnl_distribution_by_trades))

    @property
    def t_test_p_value(self) -> float:
        """
        P-value for the Student t-test, if the backtest generates PnLs that are consistently larger than zero.

        Returns
        -------
        p_value : float
            P-value for the test (without rounding).
        """
        return scipy.stats.t.sf(self.t_statistic, df=len(self.pnl_distribution_by_trades) - 1)

    @property
    def t_test_result_significant(self, significance: float = 0.05) -> bool:
        """
        Result for the Student t-test, if the backtest generates PnLs that are consistently larger than zero.

        Returns
        -------
        result : bool
            True, if the difference is significant. False otherwise.
        """
        if self.t_test_p_value <= significance:
            return True
        else:
            return False

    @abstractmethod
    def backtest(self, days_strategy: int, use_fixed_vol: bool = False) -> None:
        """
        Backtesting process that generates strategy PnL and saves the time series to self._backtest_pnl array.
        Method that should be implemented in the child class.

        Parameters
        ----------
            days_strategy : int
                days for backtesting (time till maturity of the bought option)
            use_fixed_vol : bool
                determines, whether volatility for delta-hedging will be fixed in T0 or recalculated by shifting window

        Returns
        -------
        None
        """
        self._backtest_pnl = []
        pass

    @abstractmethod
    def trading_strategy(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                         backtest_data: List[Tuple[dt.datetime, float]],
                         fixed_vol: Union[float, None] = None) -> Tuple[float, float]:
        """
        Conducts the process of trading strategy simulation for one specific time interval (one iteration of backtest).
        It's the core module of the Backtesting library, one of two that should be amended while creating a child class.

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        trading_strategy_cost, transaction_cost : tuple
            Trading strategy cost for the specified process and transaction costs paid (separately).
        """
        return 0, 0


class BacktesterOffshoreOnshore(Backtester):
    """
    This Backtester is used for testing several hypotheses of local vs offshore volatility (without correlation trade).
    Creates basic features for further strategies testing (will create child class that implement only strategy itself).

    Methods
    -------
    get_onshore_prices():
        Returns dict of local market historical prices for specified asset, using self.local_price_source
    get_offshore_prices():
        Returns dict of foreign market historical prices for specified asset, using self.foreign_price_source
    if_buy_onshore_sell_offshore():
        Decision rule for a trade - determines the market, where vol is bought vs one, where vol is sold
    trading_strategy():
        Conducts the process of trading strategy simulation for one specific time interval (one iteration of backtest)
    backtest():
        Backtests the strategy of trading lower realized vol versus higher realized vol
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hist_vols = []
        self.backtesting_dates = []
        self.opts_to_compare = []
        self.delta_hedge_to_compare = []
        self.delta_hedge_length = []

    def get_onshore_prices(self) -> Dict[dt.datetime, float]:
        """
        Onshore prices of the asset.

        Returns
        -------
        asset_prices : dict
            Onshore prices {timestamp: price} dict for the asset that is being tested.
        """
        return get_asset_prices(path=PATH, asset=self.asset, price_source=self.onshore_price_source,
                                delta_seconds=self.delta_seconds)

    def get_offshore_prices(self) -> Dict[dt.datetime, float]:
        """
        Offshore prices of the asset.

        Returns
        -------
        asset_prices : dict
            Offshore prices {timestamp: price} dict for the asset that is being tested.
        """
        return get_asset_prices(path=PATH, asset=self.asset, price_source=self.offshore_price_source,
                                delta_seconds=self.delta_seconds)

    @staticmethod
    def if_buy_onshore_sell_offshore(vol_diff_significance: float = 0, **kwargs) -> Union[bool, None]:
        """
        Provides simple decision rule for choosing, where the option should be bought or sold.
        If offshore volatility is larger by vol_diff_significance or more, we should buy locally and sell on offshore.

        Parameters
        ----------
            vol_diff_significance : float
                difference between vol_onshore and vol_offshore that is considered to be significant

        Returns
        -------
        decision_buy_offshore_sell_onshore : bool or None
            Decision for the backtested trade.
        """
        vol_onshore = kwargs['vol_onshore']
        vol_offshore = kwargs['vol_offshore']

        if vol_offshore - vol_onshore > vol_diff_significance:
            return True
        elif vol_offshore - vol_onshore < -vol_diff_significance:
            return False
        else:
            return None

    @abstractmethod
    def trading_strategy(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                         backtest_data: List[Tuple[dt.datetime, float]],
                         fixed_vol: float = None) -> Tuple[float, float]:
        """
        Conducts the process of trading strategy simulation for one specific time interval (one iteration of backtest).
        Should be inherited further.

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        trading_strategy_cost, transaction_cost : tuple
            Trading strategy cost for the specified process and transaction costs paid (separately).
        """
        return 0, 0

    def backtest(self, days_strategy: int, use_fixed_vol: bool = False, vol_diff_significance: float = 0) -> None:
        """
        Backtests the strategy of trading lower realized vol versus higher realized vol.
        Uses trading_strategy() method for generating cash cost of on iteration of the strategy
        (only locally or only on offshore) and if_buy_onshore_sell_offshore() method as decision rule.

        Parameters
        ----------
            days_strategy : int
                days for backtesting (time till maturity of the bought option)
            use_fixed_vol : bool
                determines, whether volatility for delta-hedging will be fixed in T0 or recalculated by shifting window
            vol_diff_significance : float
                difference between onshore-offshore vol that is considered to be significant, passed into decision rule

        Returns
        -------
        None
        """
        # Reinitialize stored arrays, in case backtest was already conducted previously
        self._backtest_pnl = []
        self.hist_vols = []
        self.backtesting_dates = []
        self.opts_to_compare = []
        self.delta_hedge_to_compare = []
        self.delta_hedge_length = []

        days_backtesting_period = (self.end - self.start).days
        onshore_prices = self.get_onshore_prices()
        offshore_prices = self.get_offshore_prices()

        # E.g., for hourly data will set minutes, seconds and microseconds to zero (returns tuple)
        onshore_prices = self._standardize_price_data(onshore_prices)
        offshore_prices = self._standardize_price_data(offshore_prices)

        # Get sorted by date list of tuples
        onshore_prices = sorted(onshore_prices, key=lambda x: x[0])
        offshore_prices = sorted(offshore_prices, key=lambda x: x[0])

        # Get unique set of dates, where at least one datapoint is present
        onshore_available_days = sorted(set([key.date() for key in [x[0] for x in onshore_prices]]))
        offshore_available_days = sorted(set([key.date() for key in [x[0] for x in offshore_prices]]))

        # Iterate over all available days - change the date of start for backtesting (no clustering of PnL)
        for t in range(1, days_backtesting_period - days_strategy):
            # Get date of testing as first available date + t days, iterate over t
            date_start = self.start + dt.timedelta(days=t)
            date_end = date_start + dt.timedelta(days=days_strategy)

            # Check if both assets were trading on this day (if not => just do not backtest from this day)
            if (date_start.date() in onshore_available_days) and (date_start.date() in offshore_available_days):
                # Get datasets for testing
                onshore_dataset = self._get_backtesting_dataset(onshore_prices, date_start, date_end)
                offshore_dataset = self._get_backtesting_dataset(offshore_prices, date_start, date_end)

                # Get starting spots (day of starting the backtest)
                spot_onshore_start = onshore_dataset[0][1]
                spot_offshore_start = offshore_dataset[0][1]

                # Get realized vol of days_strategy before entering the backtest
                vol_onshore = self._get_realized_vol(source=self.onshore_price_source,
                                                     date_start=date_start - dt.timedelta(days=days_strategy+1),
                                                     date_end=date_start)
                vol_offshore = self._get_realized_vol(source=self.offshore_price_source,
                                                      date_start=date_start - dt.timedelta(days=days_strategy+1),
                                                      date_end=date_start)

                # Get results of delta-hedging onshore and offshore option replication
                if use_fixed_vol:
                    # If using fixed volatility, specify the realized vol at the moment of initializing the option
                    pnl_onshore, trans_cost_onshore = self.trading_strategy(price_source=self.onshore_price_source,
                                                                            start=date_start, end=date_end,
                                                                            spot_start=spot_onshore_start,
                                                                            backtest_data=onshore_dataset,
                                                                            fixed_vol=vol_onshore)
                    pnl_offshore, trans_cost_offshore = self.trading_strategy(price_source=self.offshore_price_source,
                                                                              start=date_start, end=date_end,
                                                                              spot_start=spot_offshore_start,
                                                                              backtest_data=offshore_dataset,
                                                                              fixed_vol=vol_offshore)
                else:
                    # Else recalculate the realized vol each day (inside the delta_hedge_process, as vol_used=None)
                    pnl_onshore, trans_cost_onshore = self.trading_strategy(price_source=self.onshore_price_source,
                                                                            start=date_start, end=date_end,
                                                                            spot_start=spot_onshore_start,
                                                                            backtest_data=onshore_dataset,
                                                                            fixed_vol=None)
                    pnl_offshore, trans_cost_offshore = self.trading_strategy(price_source=self.offshore_price_source,
                                                                              start=date_start, end=date_end,
                                                                              spot_start=spot_offshore_start,
                                                                              backtest_data=offshore_dataset,
                                                                              fixed_vol=None)

                total_transaction_cost = trans_cost_onshore + trans_cost_offshore

                self.hist_vols.append((vol_onshore, vol_offshore))

                decision = self.if_buy_onshore_sell_offshore(vol_diff_significance=vol_diff_significance,
                                                             vol_onshore=vol_onshore, vol_offshore=vol_offshore)

                if pnl_onshore != 0 and pnl_offshore != 0:
                    if decision is not None:
                        if decision:
                            # PnL is cost of more expensive option minus cost of less expensive one
                            # Therefore, need to subtract cost of lower vol market from the cost of higher vol one
                            # Spread-paid is a commission => just subtract, as paid on both sides symmetrically
                            pnl = pnl_onshore - pnl_offshore - total_transaction_cost
                        else:
                            pnl = pnl_offshore - pnl_onshore - total_transaction_cost

                        # Append to the list of trade dates (for graphs)
                        self.backtesting_dates.append(date_start)
                        self._backtest_pnl.append(pnl)


class BacktesterDeltaHedgePnL(Backtester):
    """
    This Backtester is used for testing several hypotheses of local vs offshore volatility (without correlation trade).
    Creates basic features for further strategies testing (will create child class that implement only strategy itself).

    Methods
    -------
    get_onshore_prices():
        Returns dict of local market historical prices for specified asset, using self.local_price_source
    get_offshore_prices():
        Returns dict of foreign market historical prices for specified asset, using self.foreign_price_source
    if_buy_onshore_sell_offshore():
        Decision rule for a trade - determines the market, where vol is bought vs one, where vol is sold
    trading_strategy():
        Conducts the process of trading strategy simulation for one specific time interval (one iteration of backtest)
    backtest():
        Backtests the strategy of trading lower realized vol versus higher realized vol
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hist_vols = []
        self.backtesting_dates = []
        self.opts_to_compare = []
        self.delta_hedge_to_compare = []
        self.delta_hedge_length = []

    def get_prices(self) -> Dict[dt.datetime, float]:
        """
        Onshore prices of the asset.

        Returns
        -------
        asset_prices : dict
            Onshore prices {timestamp: price} dict for the asset that is being tested.
        """
        return get_asset_prices(path=PATH, asset=self.asset, price_source=self.onshore_price_source,
                                delta_seconds=self.delta_seconds)

    def trading_strategy(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                         backtest_data: List[Tuple[dt.datetime, float]],
                         fixed_vol: float = None) -> float:
        """
        Trading strategy used is the dynamic replication of the option with given params by the delta-hedging.

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        opt_cost, transaction_cost : tuple
            option cost for the specified option and transaction costs paid (separately)
        """
        return self._simulate_dynamic_delta_hedge(price_source, spot_start, start, end, backtest_data, fixed_vol)

    def _simulate_dynamic_delta_hedge(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                                      backtest_data: List[Tuple[dt.datetime, float]],
                                      fixed_vol: Union[float, None] = None) -> float:
        """
        Provides simulation of the delta hedge process on given prices for an option, specified by the backtest params.
        Returns option cost and transaction costs paid (separately).

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        opt_cost, transaction_cost : tuple
            option cost for the specified option and transaction costs paid (separately)
        """
        # First delta of the position is exactly zero => first trade would be on the full delta of the option
        delta_old = 0

        # Initialize costs (zero before the first trade)
        cash_flows_sum = 0
        rf_paid = 0
        spread_paid = 0

        # Will charge rf rate for daily residual of delta only (intraday trades are not used for funding cost)
        daily_rf_difference = (self.rf_second_ccy - self.rf_base_ccy) / 365
        rf_accumulation_start = start

        delta_hedge_points = 0

        spot_last_hedged = 0
        opt_value_path = []
        # Now iterate over available points for this backtesting period => will generate PnL[t]
        for point in backtest_data[:-1]:
            curr_date, spot = point

            # Get dates of the realized vol for delta calculation (for last n days, where n = days_strategy)
            vol_date_start = curr_date - (end - start)
            vol_date_end = curr_date

            if fixed_vol is None:
                # Calculate realized vols over last days_strategy
                vol_realized = self._get_realized_vol(source=price_source, date_start=vol_date_start,
                                                      date_end=vol_date_end)
            else:
                # If vol_used is specified, will use it for delta_hedging process
                vol_realized = fixed_vol

            # Calculate time until maturity
            till_maturity = (end - curr_date) / dt.timedelta(days=252)

            # Specify option parameters
            # Rf rate = difference (subtract base from second - e.g., subtract CNH rate from RUB rate)
            opt = {'time_till_maturity': till_maturity, 'spot': spot, 'initial_spot': spot_start,
                   'risk_free_rate': self.rf_second_ccy - self.rf_base_ccy, 'strike': spot_start,
                   'volatility': vol_realized}

            # Initialize options objects from class EuropeanCall
            call_obj = EuropeanCall(**opt)
            # For all days inside period support delta of the portfolio equal to the corresponding option
            delta = call_obj.delta

            opt_value_path.append(call_obj.price)

            # Charge daily rf rate, only if timedelta is larger than 1 day
            if curr_date - rf_accumulation_start >= dt.timedelta(days=1):
                # Add daily risk-free rate paid / received (without compounding, as rf == funding)
                rf_paid += self.notional * delta_old * daily_rf_difference
                rf_accumulation_start = curr_date

            # Check that vol point is available
            if (not np.isnan(vol_realized)) and (vol_realized != 0):
                if not np.isnan(delta - delta_old):
                    delta_hedge_points += 1
                    # Get cash cost for the delta_hedge
                    delta_hedge_cost = -self.notional * (delta - delta_old) * spot
                    # Get spread paid for hedging the delta difference
                    spread = abs(self.notional * (delta - delta_old) * self.onshore_spread)

                    # print(delta, delta_old)
                    # print(self.notional * (delta - delta_old), spread)

                    # Enter the trade to hedge delta, only if delta difference is large enough =>
                    # => need E(loss unhedged position) >= spread => need E(gamma/2 * (dS)**2) >= spread =>
                    # => as sigma ** 2 = E(r**2) - E(r)**2, we have (gamma / 2 * sigma **2) >= spread
                    # Sigma should be taken for the specified time interval => use delta seconds
                    variance_delta_time = vol_realized ** 2 / (252 * 9 * 60 * 60 / self.delta_seconds)
                    variance_in_dollars = spot ** 2 * variance_delta_time

                    expected_loss_per_option = call_obj.gamma / 2 * (variance_in_dollars + spot - spot_last_hedged)
                    if self.notional / spot_start * expected_loss_per_option >= spread:
                        # Dynamically calculate delta-hedge cost = PnL on the option
                        # Same as calculating value of the portfolio = [delta * spot + risk-free asset]
                        cash_flows_sum += delta_hedge_cost
                        # Add spread-paid (specified as % of notional traded)
                        spread_paid += spread

                        # Reinitialize delta and save last hedged delta (as already hedged)
                        delta_old = delta
                        spot_last_hedged = spot

        # At the last date we need to calculate PnL => unwind FX Spot position in full and get PnL
        # In general it is exactly the same as just calculate PnL as [delta * (spot[t] / spot[0] - 1)]
        cash_flows_sum += self.notional * delta_old * backtest_data[-1][1]
        spread_paid += abs(self.notional * delta_old * self.onshore_spread)

        # Option cost = PnL on spot dynamic delta replication + risk-free rate paid
        opt_cost = cash_flows_sum + rf_paid

        self.opts_to_compare.append(self.notional * (max(spot - spot_start, 0) / spot_start - opt_value_path[0]))
        self.delta_hedge_length.append(delta_hedge_points)
        self.delta_hedge_to_compare.append(opt_cost)
        transaction_cost = spread_paid

        return opt_cost - transaction_cost

    def backtest(self, days_strategy: int, use_fixed_vol: bool = False, vol_diff_significance: float = 0) -> None:
        """
        Backtests the strategy of trading lower realized vol versus higher realized vol.
        Uses trading_strategy() method for generating cash cost of on iteration of the strategy
        (only locally or only on offshore) and if_buy_onshore_sell_offshore() method as decision rule.

        Parameters
        ----------
            days_strategy : int
                days for backtesting (time till maturity of the bought option)
            use_fixed_vol : bool
                determines, whether volatility for delta-hedging will be fixed in T0 or recalculated by shifting window
            vol_diff_significance : float
                difference between onshore-offshore vol that is considered to be significant, passed into decision rule

        Returns
        -------
        None
        """
        # Reinitialize stored arrays, in case backtest was already conducted previously
        self._backtest_pnl = []
        self.hist_vols = []
        self.backtesting_dates = []
        self.opts_to_compare = []
        self.delta_hedge_to_compare = []
        self.delta_hedge_length = []

        days_backtesting_period = (self.end - self.start).days
        onshore_prices = self.get_prices()

        # E.g., for hourly data will set minutes, seconds and microseconds to zero (returns tuple)
        onshore_prices = self._standardize_price_data(onshore_prices)

        # Get sorted by date list of tuples
        onshore_prices = sorted(onshore_prices, key=lambda x: x[0])

        # Get unique set of dates, where at least one datapoint is present
        onshore_available_days = sorted(set([key.date() for key in [x[0] for x in onshore_prices]]))

        # Iterate over all available days - change the date of start for backtesting (no clustering of PnL)
        for t in range(1, days_backtesting_period - days_strategy):
            # Get date of testing as first available date + t days, iterate over t
            date_start = self.start + dt.timedelta(days=t)
            date_end = date_start + dt.timedelta(days=days_strategy)

            # Check if both assets were trading on this day (if not => just do not backtest from this day)
            if date_start.date() in onshore_available_days:
                # Get datasets for testing
                onshore_dataset = self._get_backtesting_dataset(onshore_prices, date_start, date_end)

                # Get starting spots (day of starting the backtest)
                spot_onshore_start = onshore_dataset[0][1]

                # Get realized vol of days_strategy before entering the backtest
                vol_onshore = self._get_realized_vol(source=self.onshore_price_source,
                                                     date_start=date_start - dt.timedelta(days=days_strategy+1),
                                                     date_end=date_start)

                # Get results of delta-hedging onshore and offshore option replication
                if use_fixed_vol:
                    # If using fixed volatility, specify the realized vol at the moment of initializing the option
                    pnl = self.trading_strategy(price_source=self.onshore_price_source, start=date_start, end=date_end,
                                                spot_start=spot_onshore_start, backtest_data=onshore_dataset,
                                                fixed_vol=vol_onshore)
                else:
                    # Else recalculate the realized vol each day (inside the delta_hedge_process, as vol_used=None)
                    pnl = self.trading_strategy(price_source=self.onshore_price_source, start=date_start, end=date_end,
                                                spot_start=spot_onshore_start, backtest_data=onshore_dataset,
                                                fixed_vol=None)

                # Check that data was available
                if pnl != 0:
                    # PnL is cost of more expensive option minus cost of less expensive one
                    # Therefore, need to subtract cost of lower vol market from the cost of higher vol one
                    # Spread-paid is a commission => just subtract, as paid on both sides symmetrically
                    self._backtest_pnl.append(pnl)
                    # Append to the list of trade dates (for graphs)
                    self.backtesting_dates.append(date_start)


class FixedLevelStrategy(BacktesterOffshoreOnshore):
    """
    STRATEGY #1 ("Stupid"):
    At T0 fix the spot level, buy at this level notional. If the spot crosses the fixed level, then sell.
    Calculate PnL at Tx, when the strategy matures.

    ...

    Methods
    -------
    if_buy_onshore_sell_offshore():
        Decision rule for a trade - if offshore vol is larger than onshore, then returns True
    trading_strategy():
        Conducts the process of trading strategy simulation via "fixed level rule"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trading_strategy(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                         backtest_data: List[Tuple[dt.datetime, float]],
                         fixed_vol: float = None) -> Tuple[float, float]:
        """
        Provides simulation of the delta hedge process on given prices for an option, specified by the backtest params.
        Returns option cost and transaction costs paid (separately).

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        opt_cost, transaction_cost : tuple
            option cost for the specified option and transaction costs paid (separately)
        """

        # Initialize costs (zero before the first trade)
        cost = 0
        rf_paid = 0
        spread_paid = 0

        # Will charge rf rate for daily residual of delta only (intraday trades are not used for funding cost)
        daily_rf_difference = (self.rf_second_ccy - self.rf_base_ccy) / 365
        rf_accumulation_start = start

        start_date, start_spot = backtest_data[0]

        position = self.notional
        # Now iterate over available points for this backtesting period => will generate PnL[t]
        for point in backtest_data[1:-1]:
            curr_date, spot = point

            spread = abs(self.notional * self.onshore_spread)
            if spot >= start_spot and position == 0:
                position = self.notional
                cost += self.notional * spot
                spread_paid += spread
            elif spot < start_spot and position > 0:
                position = 0
                cost += -self.notional * spot
                spread_paid += spread

            # Charge daily rf rate, only if timedelta is larger than 1 day
            if curr_date - rf_accumulation_start >= dt.timedelta(days=1):
                # Add daily risk-free rate paid / received (without compounding, as rf == funding)
                rf_paid += position * daily_rf_difference
                rf_accumulation_start = curr_date

        # At the last date we need to calculate PnL => unwind FX Spot position in full and get PnL
        # In general it is exactly the same as just calculate PnL as [delta * (spot[t] / spot[0] - 1)]
        cost += -position * backtest_data[-1][1]
        spread_paid += abs(-position * self.onshore_spread)

        # Option cost = PnL on spot dynamic delta replication + risk-free rate paid
        opt_cost = cost + rf_paid
        transaction_cost = spread_paid

        return opt_cost, transaction_cost


class DynamicDeltaHedgeStrategy(BacktesterOffshoreOnshore):
    """
    STRATEGY #3 (Advanced):
    Create an option on onshore and offshore by dynamic delta-hedge
    (vol fixed at the option creation moment or recalculated by shifting window of n*frequency=days_strategy points).

    Methods
    -------
    trading_strategy():
        Applied _simulate_delta_hedge() method, passes it into parent
    _simulate_delta_hedge():
        Provides simulation of the delta hedge process on given prices for an option, specified by the backtest params
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trading_strategy(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                         backtest_data: List[Tuple[dt.datetime, float]],
                         fixed_vol: float = None) -> Tuple[float, float]:
        """
        Trading strategy used is the dynamic replication of the option with given params by the delta-hedging.

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        opt_cost, transaction_cost : tuple
            option cost for the specified option and transaction costs paid (separately)
        """
        return self._simulate_dynamic_delta_hedge(price_source, spot_start, start, end, backtest_data, fixed_vol)

    def _simulate_dynamic_delta_hedge(self, price_source: str, spot_start: float, start: dt.datetime, end: dt.datetime,
                                      backtest_data: List[Tuple[dt.datetime, float]],
                                      fixed_vol: Union[float, None] = None) -> Tuple[float, float]:
        """
        Provides simulation of the delta hedge process on given prices for an option, specified by the backtest params.
        Returns option cost and transaction costs paid (separately).

        Parameters
        ----------
            price_source : str
                source for prices, for which the option is being replicated
            spot_start : float
                starting spot value
            start : datetime.datetime
                first date of the period window
            end : datetime.datetime
                last date of the period window
            backtest_data : list
                price data for specified backtesting interval (will iterate only over this data)
            fixed_vol : float
                specify volatility for pricing the option. Use None to calculate realized vol by a shifting window

        Returns
        -------
        opt_cost, transaction_cost : tuple
            option cost for the specified option and transaction costs paid (separately)
        """
        # First delta of the position is exactly zero => first trade would be on the full delta of the option
        delta_old = 0

        # Initialize costs (zero before the first trade)
        cash_flows_sum = 0
        rf_paid = 0
        spread_paid = 0

        # Will charge rf rate for daily residual of delta only (intraday trades are not used for funding cost)
        daily_rf_difference = (self.rf_second_ccy - self.rf_base_ccy) / 365
        rf_accumulation_start = start

        delta_hedge_points = 0

        spot_last_hedged = 0
        opt_value_path = []
        # Now iterate over available points for this backtesting period => will generate PnL[t]
        for point in backtest_data[:-1]:
            curr_date, spot = point

            # Get dates of the realized vol for delta calculation (for last n days, where n = days_strategy)
            vol_date_start = curr_date - (end - start)
            vol_date_end = curr_date

            if fixed_vol is None:
                # Calculate realized vols over last days_strategy
                vol_realized = self._get_realized_vol(source=price_source, date_start=vol_date_start,
                                                      date_end=vol_date_end)
            else:
                # If vol_used is specified, will use it for delta_hedging process
                vol_realized = fixed_vol

            # Calculate time until maturity
            till_maturity = (end - curr_date) / dt.timedelta(days=252)

            # Specify option parameters
            # Rf rate = difference (subtract base from second - e.g., subtract CNH rate from RUB rate)
            opt = {'time_till_maturity': till_maturity, 'spot': spot, 'initial_spot': spot_start,
                   'risk_free_rate': self.rf_second_ccy - self.rf_base_ccy, 'strike': spot_start,
                   'volatility': vol_realized}

            # Initialize options objects from class EuropeanCall
            call_obj = EuropeanCall(**opt)
            # For all days inside period support delta of the portfolio equal to the corresponding option
            delta = call_obj.delta

            opt_value_path.append(call_obj.price)

            # Charge daily rf rate, only if timedelta is larger than 1 day
            if curr_date - rf_accumulation_start >= dt.timedelta(days=1):
                # Add daily risk-free rate paid / received (without compounding, as rf == funding)
                rf_paid += self.notional * delta_old * daily_rf_difference
                rf_accumulation_start = curr_date

            # Check that vol point is available
            if (not np.isnan(vol_realized)) and (vol_realized != 0):
                if not np.isnan(delta - delta_old):
                    delta_hedge_points += 1
                    # Get cash cost for the delta_hedge
                    delta_hedge_cost = -self.notional * (delta - delta_old) * spot
                    # Get spread paid for hedging the delta difference
                    spread = abs(self.notional * (delta - delta_old) * self.onshore_spread)

                    # print(delta, delta_old)
                    # print(self.notional * (delta - delta_old), spread)

                    # Enter the trade to hedge delta, only if delta difference is large enough =>
                    # => need E(loss unhedged position) >= spread => need E(gamma/2 * (dS)**2) >= spread =>
                    # => as sigma ** 2 = E(r**2) - E(r)**2, we have (gamma / 2 * sigma **2) >= spread
                    # Sigma should be taken for the specified time interval => use delta seconds
                    variance_delta_time = vol_realized ** 2 / (252 * 9 * 60 * 60 / self.delta_seconds)
                    variance_in_dollars = spot ** 2 * variance_delta_time

                    expected_loss_per_option = call_obj.gamma / 2 * (variance_in_dollars + spot - spot_last_hedged)
                    if self.notional / spot_start * expected_loss_per_option >= spread:
                        # Dynamically calculate delta-hedge cost = PnL on the option
                        # Same as calculating value of the portfolio = [delta * spot + risk-free asset]
                        cash_flows_sum += delta_hedge_cost
                        # Add spread-paid (specified as % of notional traded)
                        spread_paid += spread

                        # Reinitialize delta and save last hedged delta (as already hedged)
                        delta_old = delta
                        spot_last_hedged = spot

        # At the last date we need to calculate PnL => unwind FX Spot position in full and get PnL
        # In general it is exactly the same as just calculate PnL as [delta * (spot[t] / spot[0] - 1)]
        cash_flows_sum += self.notional * delta_old * backtest_data[-1][1]
        spread_paid += abs(self.notional * delta_old * self.onshore_spread)

        # Option cost = PnL on spot dynamic delta replication + risk-free rate paid
        opt_cost = cash_flows_sum + rf_paid

        self.opts_to_compare.append(self.notional * (max(spot - spot_start, 0) / spot_start - opt_value_path[0]))
        self.delta_hedge_length.append(delta_hedge_points)
        self.delta_hedge_to_compare.append(opt_cost)
        transaction_cost = spread_paid

        return opt_cost, transaction_cost
