"""
Backtest Engine
===============
Backtest portfolio strategies based on ranking model.

Features:
  - Daily rebalancing
  - Long/short portfolios
  - Sharpe ratio, drawdown, turnover
  - Hit rate (did ranks predict actual returns?)
  - Transaction costs
  - Regime filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtest multi-stock ranking strategies.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,  # Annual
        transaction_cost: float = 0.001,  # 10 bps per trade
        slippage: float = 0.0005,  # 5 bps
    ):
        """
        Initialize backtest engine.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            transaction_cost: Cost per transaction (0.001 = 0.1%)
            slippage: Price slippage per trade
        """
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def backtest_ranking_strategy(
        self,
        returns_df: pd.DataFrame,
        ranks_df: pd.DataFrame,
        long_n: int = 5,
        short_n: int = 5,
        rebalance_freq: str = "D",  # Daily
        leverage: float = 1.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        holding_period: int = 1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Backtest a long/short ranking strategy.
        """
        df = returns_df.merge(ranks_df, on=["date", "ticker"], how="inner")
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        dates = sorted(df["date"].unique())
        trade_dates = self._schedule_rebalance_dates(dates, rebalance_freq, holding_period)

        portfolio_returns = []
        benchmark_returns = []
        dates_list = []
        prev_holdings = {"long": [], "short": []}

        for i, date in enumerate(trade_dates):
            date_df = df[df["date"] == date]
            if date_df.empty:
                continue

            ranked_today = date_df.sort_values("rank")
            longs = ranked_today.head(long_n)["ticker"].tolist()
            shorts = ranked_today.tail(short_n)["ticker"].tolist()

            next_dates = [d for d in dates if d > date]
            if not next_dates:
                break
            next_date = next_dates[0]
            next_date_df = df[df["date"] == next_date]

            benchmark = next_date_df["return"].mean() if not next_date_df.empty else 0.0
            benchmark_returns.append(float(np.nan_to_num(benchmark, 0.0)))

            long_returns = self._apply_trade_constraints(next_date_df, longs, stop_loss_pct, take_profit_pct, is_short=False)
            short_returns = self._apply_trade_constraints(next_date_df, shorts, stop_loss_pct, take_profit_pct, is_short=True)

            gross_return = 0.5 * long_returns + 0.5 * short_returns
            turnover = self._compute_turnover(prev_holdings["long"], longs, prev_holdings["short"], shorts)
            transaction_cost_loss = turnover * (transaction_cost + slippage) * leverage
            net_return = gross_return * leverage - transaction_cost_loss

            portfolio_returns.append(float(np.nan_to_num(net_return, 0.0)))
            dates_list.append(next_date)
            prev_holdings = {"long": longs, "short": shorts}

        returns_series = pd.Series(portfolio_returns, index=dates_list)
        metrics = self._compute_metrics(returns_series)

        return {
            "dates": dates_list,
            "returns": returns_series.tolist(),
            "benchmark": benchmark_returns,
            "metrics": metrics,
        }

    def backtest_long_only(
        self,
        returns_df: pd.DataFrame,
        ranks_df: pd.DataFrame,
        long_n: int = 5,
        rebalance_freq: str = "D",
        leverage: float = 1.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        holding_period: int = 1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Backtest long-only strategy."""
        df = returns_df.merge(ranks_df, on=["date", "ticker"], how="inner")
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        dates = sorted(df["date"].unique())
        trade_dates = self._schedule_rebalance_dates(dates, rebalance_freq, holding_period)

        portfolio_returns = []
        benchmark_returns = []
        dates_list = []
        prev_holdings = {"long": []}

        for date in trade_dates:
            date_df = df[df["date"] == date]
            if date_df.empty:
                continue

            ranked_today = date_df.sort_values("rank")
            longs = ranked_today.head(long_n)["ticker"].tolist()

            next_dates = [d for d in dates if d > date]
            if not next_dates:
                break
            next_date = next_dates[0]
            next_date_df = df[df["date"] == next_date]

            benchmark = next_date_df["return"].mean() if not next_date_df.empty else 0.0
            benchmark_returns.append(float(np.nan_to_num(benchmark, 0.0)))

            long_returns = self._apply_trade_constraints(next_date_df, longs, stop_loss_pct, take_profit_pct, is_short=False)
            turnover = self._compute_turnover(prev_holdings.get("long", []), longs, [], [])
            cost = turnover * (transaction_cost + slippage) * leverage
            net_return = long_returns * leverage - cost

            portfolio_returns.append(float(np.nan_to_num(net_return, 0.0)))
            dates_list.append(next_date)
            prev_holdings = {"long": longs}

        returns_series = pd.Series(portfolio_returns, index=dates_list)
        metrics = self._compute_metrics(returns_series)

        return {
            "dates": dates_list,
            "returns": returns_series.tolist(),
            "benchmark": benchmark_returns,
            "metrics": metrics,
        }

    def _schedule_rebalance_dates(self, dates: List[datetime], frequency: str, holding_period: int) -> List[datetime]:
        if frequency == "W":
            return [d for i, d in enumerate(dates) if i % 5 == 0]
        if frequency == "M":
            return [d for d in dates if d.day == 1]
        return dates[::holding_period]

    def _apply_trade_constraints(
        self,
        next_date_df: pd.DataFrame,
        tickers: List[str],
        stop_loss_pct: float,
        take_profit_pct: float,
        is_short: bool = False,
    ) -> float:
        if not tickers:
            return 0.0
        position_returns = next_date_df[next_date_df["ticker"].isin(tickers)]["return"].tolist()
        if not position_returns:
            return 0.0

        capped_returns = []
        for rtn in position_returns:
            trade_return = -rtn if is_short else rtn
            if stop_loss_pct > 0:
                trade_return = max(trade_return, -stop_loss_pct)
            if take_profit_pct > 0:
                trade_return = min(trade_return, take_profit_pct)
            capped_returns.append(trade_return)

        average_return = float(np.nan_to_num(np.mean(capped_returns), 0.0))
        return average_return

    def _compute_turnover(
        self,
        prev_long: List[str],
        curr_long: List[str],
        prev_short: List[str],
        curr_short: List[str],
    ) -> float:
        """
        Compute portfolio turnover.
        """
        long_exits = len(set(prev_long) - set(curr_long))
        long_entries = len(set(curr_long) - set(prev_long))
        
        short_exits = len(set(prev_short) - set(curr_short))
        short_entries = len(set(curr_short) - set(prev_short))
        
        total_changes = long_exits + long_entries + short_exits + short_entries
        return total_changes / max(len(prev_long) + len(prev_short), 1) / 2

    def _compute_metrics(self, returns_series: pd.Series) -> Dict:
        """
        Compute performance metrics.

        """
        if len(returns_series) < 2:
            return {}

        # Daily metrics
        daily_returns = returns_series.values
        
        # Annual metrics (assume 252 trading days)
        annual_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
        annual_vol = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (daily_returns > 0).mean()
        
        # Hit rate (positive returns predict positive next returns)
        if len(returns_series) > 1:
            actual_direction = (daily_returns[1:] > 0).astype(float)
            pred_direction = (daily_returns[:-1] > 0).astype(float)
            hit_rate = (actual_direction == pred_direction).mean()
        else:
            hit_rate = 0

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "hit_rate": hit_rate,
            "total_return": cumulative[-1] - 1 if len(cumulative) > 0 else 0,
            "trade_count": len(returns_series),
        }

    def regime_filter(
        self,
        portfolio_returns: pd.Series,
        macro_regimes: pd.Series,
        filter_rules: Optional[Dict] = None,
    ) -> pd.Series:
        """
        Apply regime-based filters to strategy.
        
        Example:
          - if yields spike: reduce gross leverage
          - if VIX > 30: reduce gross leverage
          - if gold volatility high: reduce gross leverage
        
        Args:
            portfolio_returns: Original returns
            macro_regimes: Macro regime for each date
            filter_rules: {regime: scale_factor}
        
        Returns:
            Adjusted returns
        """
        if filter_rules is None:
            filter_rules = {
                "risk_off": 0.5,  # 50% of position
                "neutral": 1.0,
                "risk_on": 1.0,
            }

        adjusted_returns = portfolio_returns.copy()
        
        for idx, (date, returns) in enumerate(portfolio_returns.items()):
            if idx < len(macro_regimes):
                regime = macro_regimes.iloc[idx]
                scale = filter_rules.get(regime, 1.0)
                adjusted_returns.iloc[idx] = returns * scale
        
        return adjusted_returns

    def compare_strategies(
        self,
        strategies: Dict[str, Dict],  # {name: {returns, metrics}}
    ) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side.
        
        Args:
            strategies: Dictionary of strategy results
        
        Returns:
            DataFrame with metrics for each strategy
        """
        comparison = []
        
        for name, strategy in strategies.items():
            metrics = strategy.get("metrics", {})
            metrics["strategy"] = name
            comparison.append(metrics)
        
        return pd.DataFrame(comparison)

    def monte_carlo_analysis(
        self,
        returns_series: pd.Series,
        n_simulations: int = 1000,
        n_days: int = 252,
    ) -> Dict:
        """
        Run Monte Carlo analysis on returns.
        
        Simulates possible future paths using historical returns.
        
        Returns:
            {
                confidence_intervals: {5th, 25th, 50th, 75th, 95th percentiles},
                probability_ruin: float,
            }
        """
        daily_returns = returns_series.values
        
        # Run simulations
        simulations = []
        for _ in range(n_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(daily_returns, size=n_days, replace=True)
            path = (1 + sampled_returns).cumprod()
            simulations.append(path[-1])
        
        simulations = np.array(simulations)
        
        # Compute statistics
        percentiles = np.percentile(simulations, [5, 25, 50, 75, 95])
        
        # Probability of ruin (total loss)
        prob_ruin = (simulations < 0).mean()
        
        return {
            "confidence_intervals": {
                "p5": percentiles[0],
                "p25": percentiles[1],
                "median": percentiles[2],
                "p75": percentiles[3],
                "p95": percentiles[4],
            },
            "probability_ruin": prob_ruin,
            "mean_final_value": simulations.mean(),
            "std_final_value": simulations.std(),
        }
