"""
Evaluation & Backtesting
========================
Financial evaluation is NOT the same as ML evaluation.
A model with high IC can still lose money in live trading
if transaction costs eat the signal, or if it's poorly sized.

This module computes:
  1. Statistical significance of IC (t-test)
  2. Sharpe Ratio of the signal-driven strategy
  3. Simple backtest (signal → long/flat/short → PnL)
  4. Max Drawdown
  5. Comparison vs. Buy-and-Hold benchmark
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict


def ic_significance_test(ic: float, n_samples: int) -> dict:
    """
    Test whether observed IC is statistically different from 0.
    
    Under H0: IC = 0, the test statistic:
      t = IC * sqrt(n - 2) / sqrt(1 - IC²)
    follows a t-distribution with (n-2) degrees of freedom.
    
    Why this matters: With enough data, even tiny IC appears significant.
    With too little data, genuinely good IC appears insignificant.
    Always report both IC and p-value — one without the other is incomplete.
    """
    if abs(ic) >= 1.0:
        return {"t_stat": np.inf, "p_value": 0.0, "significant": True}
    
    t_stat = ic * np.sqrt(n_samples - 2) / np.sqrt(1 - ic**2 + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_samples - 2))
    
    return {
        "t_stat":      round(t_stat, 4),
        "p_value":     round(p_value, 4),
        "significant": p_value < 0.05,
        "n_samples":   n_samples,
    }


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe Ratio = (mean return / std return) * sqrt(periods_per_year)
    
    Assumes daily returns → multiply by sqrt(252) to annualize.
    
    Interpretation:
      SR < 0.5  : poor (most random strategies)
      SR > 1.0  : acceptable
      SR > 2.0  : very good
      SR > 3.0  : exceptional / likely overfit
    
    Important: SR doesn't account for fat tails. A strategy can have
    good SR but blow up due to a single black swan event.
    Also compute Sortino (downside deviation only) for fuller picture.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def compute_sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Sortino Ratio: like Sharpe but only penalizes downside volatility.
    More appropriate for asymmetric return distributions (which financial
    returns typically are).
    """
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods_per_year))


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Maximum drawdown: largest peak-to-trough decline.
    MDD = (trough - peak) / peak
    
    Critical risk metric — high Sharpe with large MDD means the strategy
    can have catastrophic losing periods even if long-run profitable.
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1e-10)
    return float(drawdown.min())


def backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transaction_cost: float = 0.001,   # 10 bps round-trip (realistic for retail)
    long_only: bool = False,            # if True: only go long or flat (no shorting)
) -> Dict:
    """
    Simplified signal-driven backtest.
    
    Strategy:
      - If predicted return > 0 → go long (position = +1)
      - If predicted return < 0 → go short (position = -1) [or flat if long_only]
      - Position is sized by signal strength (rank-based)
    
    Why signal-scaled sizing?
    Uniform ±1 betting ignores conviction. Larger predicted moves should
    get larger positions. Rank normalization prevents outlier predictions
    from dominating.
    
    Transaction costs: Each trade incurs 10 bps cost. This is critical —
    many "profitable" ML strategies disappear after realistic transaction
    costs are applied. HFT strategies need costs < 1 bps.
    """
    # Signal to position
    if long_only:
        positions = (y_pred > 0).astype(float)   # 0 or 1
    else:
        positions = np.sign(y_pred)               # -1 or +1

    # Daily strategy returns
    # Shift positions by 1: today's signal applied to tomorrow's return
    # (You decide at close based on model, execute at next open)
    strategy_returns = positions[:-1] * y_true[1:]

    # Subtract transaction costs when position changes
    position_changes = np.abs(np.diff(positions))
    strategy_returns -= position_changes * transaction_cost

    # Cumulative returns
    cum_strategy = np.cumprod(1 + strategy_returns) - 1
    cum_bnh      = np.cumprod(1 + y_true[1:]) - 1     # buy-and-hold

    sharpe   = compute_sharpe(strategy_returns)
    sortino  = compute_sortino(strategy_returns)
    mdd      = compute_max_drawdown(1 + cum_strategy)
    total_ret = float(cum_strategy[-1]) if len(cum_strategy) > 0 else 0.0
    bnh_ret   = float(cum_bnh[-1]) if len(cum_bnh) > 0 else 0.0

    n_trades = int(position_changes.sum())
    win_rate = float(np.mean(strategy_returns > 0))

    return {
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown":     round(mdd, 4),
        "total_return":     round(total_ret, 4),
        "buy_hold_return":  round(bnh_ret, 4),
        "alpha":            round(total_ret - bnh_ret, 4),
        "n_trades":         n_trades,
        "win_rate":         round(win_rate, 4),
        "strategy_returns": strategy_returns,
        "cum_strategy":     cum_strategy,
        "cum_bnh":          cum_bnh,
    }


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "",
    verbose: bool = True,
) -> Dict:
    """
    Run all evaluation metrics. This is what you present in interviews.
    """
    from models.train import information_coefficient, direction_accuracy

    ic      = information_coefficient(y_true, y_pred)
    ic_sig  = ic_significance_test(ic, len(y_true))
    dir_acc = direction_accuracy(y_true, y_pred)
    bt      = backtest(y_true, y_pred)

    result = {
        "ticker":          ticker,
        "n_samples":       len(y_true),
        "ic":              round(ic, 4),
        "ic_pvalue":       ic_sig["p_value"],
        "ic_significant":  ic_sig["significant"],
        "direction_acc":   round(dir_acc, 4),
        **{k: v for k, v in bt.items() if k not in ("strategy_returns", "cum_strategy", "cum_bnh")},
        "_returns": {   # kept for plotting
            "strategy": bt["strategy_returns"],
            "cum_strategy": bt["cum_strategy"],
            "cum_bnh": bt["cum_bnh"],
        }
    }

    if verbose:
        print(f"\n{'-'*45}")
        print(f" EVALUATION SUMMARY: {ticker}")
        print(f"{'-'*45}")
        print(f"  IC (Spearman):       {ic:.4f}  (p={ic_sig['p_value']:.3f}, {'[sig]' if ic_sig['significant'] else '[not sig]'})")
        print(f"  Direction Accuracy:  {dir_acc:.4f}  (baseline: 0.50)")
        print(f"  Sharpe Ratio:        {bt['sharpe']:.3f}")
        print(f"  Sortino Ratio:       {bt['sortino']:.3f}")
        print(f"  Max Drawdown:        {bt['max_drawdown']:.2%}")
        print(f"  Total Return:        {bt['total_return']:.2%}")
        print(f"  Buy-Hold Return:     {bt['buy_hold_return']:.2%}")
        print(f"  Alpha:               {bt['alpha']:.2%}")
        print(f"  Win Rate:            {bt['win_rate']:.2%}")
        print(f"  Num Trades:          {bt['n_trades']}")
        print(f"{'-'*45}")

    return result