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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Backtest a long/short ranking strategy.
        
        Args:
            returns_df: DataFrame with date, ticker, returns
            ranks_df: DataFrame with date, ticker, rank
            long_n: Number of long positions
            short_n: Number of short positions
            rebalance_freq: "D" (daily), "W" (weekly), "M" (monthly)
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            {
                returns: Series of strategy returns,
                metrics: {sharpe, max_dd, win_rate, turnover, ...}
            }
        """
        # Merge returns and ranks
        df = returns_df.merge(ranks_df, on=["date", "ticker"], how="inner")
        
        # Filter date range
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        # Group by date
        dates = sorted(df["date"].unique())
        
        # Initialize portfolio
        portfolio_returns = []
        dates_list = []
        
        prev_holdings = {"long": [], "short": []}
        
        for date in dates:
            date_df = df[df["date"] == date]
            
            # Get top longs and shorts
            ranked_today = date_df.sort_values("rank")
            
            longs = ranked_today.head(long_n)["ticker"].tolist()
            shorts = ranked_today.tail(short_n)["ticker"].tolist()
            
            # Get tomorrow's returns
            tomorrow_df = df[df["date"] > date]
            if tomorrow_df.empty:
                break
            
            next_date = tomorrow_df["date"].min()
            next_date_df = tomorrow_df[tomorrow_df["date"] == next_date]
            
            # Compute strategy return
            long_returns = next_date_df[next_date_df["ticker"].isin(longs)]["return"].mean()
            short_returns = next_date_df[next_date_df["ticker"].isin(shorts)]["return"].mean()
            
            if np.isnan(long_returns):
                long_returns = 0
            if np.isnan(short_returns):
                short_returns = 0
            
            # Portfolio return: equal weight long, equal weight short, 50/50 split
            gross_return = 0.5 * long_returns - 0.5 * short_returns
            
            # Turnover cost
            turnover = self._compute_turnover(
                prev_holdings["long"], longs,
                prev_holdings["short"], shorts
            )
            transaction_cost_loss = turnover * (self.transaction_cost + self.slippage)
            
            net_return = gross_return - transaction_cost_loss
            
            portfolio_returns.append(net_return)
            dates_list.append(next_date)
            
            prev_holdings = {"long": longs, "short": shorts}
        
        # Compute metrics
        returns_series = pd.Series(portfolio_returns, index=dates_list)
        
        metrics = self._compute_metrics(returns_series)
        
        return {
            "returns": returns_series,
            "metrics": metrics,
        }

    def backtest_long_only(
        self,
        returns_df: pd.DataFrame,
        ranks_df: pd.DataFrame,
        long_n: int = 5,
        rebalance_freq: str = "D",
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
        portfolio_returns = []
        dates_list = []
        
        for date in dates:
            date_df = df[df["date"] == date]
            ranked_today = date_df.sort_values("rank")
            
            longs = ranked_today.head(long_n)["ticker"].tolist()
            
            tomorrow_df = df[df["date"] > date]
            if tomorrow_df.empty:
                break
            
            next_date = tomorrow_df["date"].min()
            next_date_df = tomorrow_df[tomorrow_df["date"] == next_date]
            
            long_returns = next_date_df[next_date_df["ticker"].isin(longs)]["return"].mean()
            if np.isnan(long_returns):
                long_returns = 0
            
            portfolio_returns.append(long_returns)
            dates_list.append(next_date)
        
        returns_series = pd.Series(portfolio_returns, index=dates_list)
        metrics = self._compute_metrics(returns_series)
        
        return {
            "returns": returns_series,
            "metrics": metrics,
        }

    def _compute_turnover(
        self,
        prev_long: List[str],
        curr_long: List[str],
        prev_short: List[str],
        curr_short: List[str],
    ) -> float:
        """
        Compute portfolio turnover.
        
        Turnover = (outflows + inflows) / 2 / total_portfolio
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
        
        Returns:
            {sharpe, annual_return, max_drawdown, win_rate, hit_rate}
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
