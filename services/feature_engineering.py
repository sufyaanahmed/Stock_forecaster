"""
Feature Engineering Overhaul
============================
Generate comprehensive features for cross-sectional ranking.

Stock-specific features:
  - returns, volatility, momentum
  - technical indicators (RSI, MACD, Bollinger)
  - volume patterns
  - gaps

Interaction features (stock × macro):
  - return * yield_change
  - RSI * gold_momentum
  - volatility * fed_rate
  - sector effects

These create a macro-regime-aware feature set.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    include_technical: bool = True
    include_volume: bool = True
    include_gaps: bool = True
    include_interactions: bool = True
    include_sector: bool = False  # requires sector data
    lookback: int = 60


class FeatureEngineer:
    """
    Comprehensive feature generation for ranking model.
    
    NOT for single-stock prediction — these features are designed
    for relative ranking within universes.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    # ─────────────────────────────────────────────────────────────────────────
    # STOCK-SPECIFIC FEATURES
    # ─────────────────────────────────────────────────────────────────────────

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute returns at multiple horizons.
        
        Features:
          - return_1d, return_5d, return_20d
          - log_return_1d
        """
        df = df.copy()
        
        df["return_1d"] = df.groupby("ticker")["Close"].pct_change(1)
        df["return_5d"] = df.groupby("ticker")["Close"].pct_change(5)
        df["return_20d"] = df.groupby("ticker")["Close"].pct_change(20)
        
        # Log returns
        df["log_return_1d"] = np.log(df["Close"] / df.groupby("ticker")["Close"].shift(1))
        
        return df

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute realized volatility at multiple horizons.
        
        Features:
          - volatility_10, volatility_60
          - volatility_ratio (short/long)
          - log_volatility
        """
        df = df.copy()
        
        df["volatility_10"] = df.groupby("ticker")["return_1d"].rolling(10).std().values
        df["volatility_60"] = df.groupby("ticker")["return_1d"].rolling(60).std().values
        df["volatility_ratio"] = df["volatility_10"] / (df["volatility_60"] + 1e-8)
        
        return df

    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index.
        
        RSI = 100 - [100 / (1 + RS)]
        RS = avg(up) / avg(down)
        
        Range: [0, 100]
        Overbought: > 70
        Oversold: < 30
        """
        df = df.copy()
        
        def rsi_group(group):
            delta = group["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df["RSI"] = df.groupby("ticker").apply(rsi_group).values
        
        return df

    def compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(9) of MACD
        Histogram = MACD - Signal
        """
        df = df.copy()
        
        def macd_group(group):
            ema_fast = group["Close"].ewm(span=fast).mean()
            ema_slow = group["Close"].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return macd, signal, histogram
        
        macd_vals = []
        signal_vals = []
        hist_vals = []
        
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker]
            macd, signal, hist = macd_group(ticker_df)
            macd_vals.extend(macd.values)
            signal_vals.extend(signal.values)
            hist_vals.extend(hist.values)
        
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["MACD"] = macd_vals
        df["MACD_signal"] = signal_vals
        df["MACD_histogram"] = hist_vals
        
        return df

    def compute_bollinger(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Features:
          - bollinger_position: (Close - Lower) / (Upper - Lower)
            Range [0, 1]: 0 = at lower band, 1 = at upper band
          - bollinger_width: (Upper - Lower) / Middle
        """
        df = df.copy()
        
        def bollinger_group(group):
            sma = group["Close"].rolling(period).mean()
            std = group["Close"].rolling(period).std()
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            width = (upper - lower) / (sma + 1e-8)
            position = (group["Close"] - lower) / (upper - lower + 1e-8)
            position = position.clip(0, 1)  # Clamp to [0, 1]
            return position, width
        
        positions = []
        widths = []
        
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].sort_values("date")
            pos, width = bollinger_group(ticker_df)
            positions.extend(pos.values)
            widths.extend(width.values)
        
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["bollinger_position"] = positions
        df["bollinger_width"] = widths
        
        return df

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features.
        
        Features:
          - volume_ratio: current / 20-day avg
          - volume_shock: change in volume
          - volume_trend: 5-day vs 20-day average
        """
        df = df.copy()
        
        df["volume_20d_avg"] = df.groupby("ticker")["Volume"].rolling(20).mean().values
        df["volume_ratio"] = df["Volume"] / (df["volume_20d_avg"] + 1e-8)
        
        df["volume_shock"] = df.groupby("ticker")["Volume"].pct_change(1)
        
        volume_5d = df.groupby("ticker")["Volume"].rolling(5).mean().values
        volume_20d = df.groupby("ticker")["Volume"].rolling(20).mean().values
        df["volume_trend"] = volume_5d / (volume_20d + 1e-8)
        
        return df

    def compute_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Overnight gaps.
        
        Features:
          - overnight_gap: (Open - Prev Close) / Prev Close
          - gap_direction: sign of overnight gap
        
        Note: skips gracefully if 'Open' column is absent.
        """
        df = df.copy()

        if "Open" not in df.columns:
            logger.debug("'Open' column not available — gap features set to 0")
            df["overnight_gap"]  = 0.0
            df["gap_direction"]  = 0.0
            return df

        df["prev_close"] = df.groupby("ticker")["Close"].shift(1)
        df["overnight_gap"] = (df["Open"] - df["prev_close"]) / (df["prev_close"] + 1e-8)
        df["gap_direction"] = np.sign(df["overnight_gap"])

        return df

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intra-period momentum.
        
        Features:
          - momentum_5d: momentum over 5 days
          - momentum_20d: momentum over 20 days
          - momentum_ratio: short / long
        """
        df = df.copy()
        
        df["momentum_5d"] = df.groupby("ticker")["Close"].rolling(5).apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0
        ).values
        
        df["momentum_20d"] = df.groupby("ticker")["Close"].rolling(20).apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0
        ).values
        
        df["momentum_ratio"] = df["momentum_5d"] / (df["momentum_20d"] + 1e-8)
        
        return df

    def compute_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Relative strength vs universe.
        
        Features:
          - relative_strength: stock return vs mean universe return
          - outperformance: deviation from peer average
        """
        df = df.copy()
        
        # Average return by date
        df["universe_mean_return"] = df.groupby("date")["return_1d"].transform("mean")
        df["relative_strength"] = df["return_1d"] - df["universe_mean_return"]
        
        # Outperformance (standardized)
        df["outperformance"] = df.groupby("date")["return_1d"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTION FEATURES (Stock × Macro)
    # ─────────────────────────────────────────────────────────────────────────

    def add_macro_interactions(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between stock and macro data.
        
        These capture how macro regimes affect stock behavior.
        
        Examples:
          - return_1d * yield_change: growth sensitivity
          - RSI * gold_momentum: mean reversion under macro stress
          - volatility * vix_level: vol clustering
        """
        # Merge macro data by date
        result_df = stock_df.merge(macro_df, left_on="date", right_index=True, how="left")
        
        # Growth stock × yields interaction
        if "yield_change_5d" in result_df.columns:
            result_df["return_x_yield"] = result_df.get("return_1d", 0) * result_df.get("yield_change_5d", 0)
        
        # Value/momentum × gold interaction
        if "gold_momentum" in result_df.columns:
            result_df["rsi_x_gold"] = result_df.get("RSI", 0) / 50 * result_df.get("gold_momentum", 0)
        
        # Vol clustering
        if "vix_level" in result_df.columns:
            result_df["vol_x_vix"] = result_df.get("volatility_10", 0) * result_df.get("vix_level", 0) / 100
        
        # Momentum × macro trend
        if "yield_momentum" in result_df.columns:
            result_df["momentum_x_yield"] = result_df.get("momentum_5d", 0) * result_df.get("yield_momentum", 0)
        
        # Volume × risk regime
        if "risk_regime_score" in result_df.columns:
            result_df["volume_x_risk"] = result_df.get("volume_ratio", 0) * result_df.get("risk_regime_score", 0)
        
        return result_df

    # ─────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def engineer_stock_features(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """Compute all stock-specific features."""
        df = stock_df.copy()
        df = df.sort_values(["ticker", "date"])
        
        logger.info("Computing returns...")
        df = self.compute_returns(df)
        
        logger.info("Computing volatility...")
        df = self.compute_volatility(df)
        
        if self.config.include_technical:
            logger.info("Computing RSI...")
            df = self.compute_rsi(df)
            
            logger.info("Computing MACD...")
            df = self.compute_macd(df)
            
            logger.info("Computing Bollinger...")
            df = self.compute_bollinger(df)
        
        if self.config.include_volume:
            logger.info("Computing volume features...")
            df = self.compute_volume_features(df)
        
        if self.config.include_gaps:
            logger.info("Computing gaps...")
            df = self.compute_gaps(df)
        
        logger.info("Computing momentum...")
        df = self.compute_momentum(df)
        
        logger.info("Computing relative strength...")
        df = self.compute_relative_strength(df)
        
        return df

    def engineer_all_features(
        self, stock_df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        
        Args:
            stock_df: Stock OHLCV data with date, ticker, Close, Volume, target_rank
            macro_df: Macro features indexed by date (optional)
        
        Returns:
            Feature-rich DataFrame ready for modeling
        """
        df = self.engineer_stock_features(stock_df)
        
        if self.config.include_interactions and macro_df is not None:
            logger.info("Adding macro interactions...")
            df = self.add_macro_interactions(df, macro_df)
        
        return df

    def get_feature_list(self, include_target: bool = True) -> List[str]:
        """Return list of computed features."""
        features = [
            "return_1d", "return_5d", "return_20d",
            "log_return_1d",
            "volatility_10", "volatility_60", "volatility_ratio",
        ]
        
        if self.config.include_technical:
            features.extend(["RSI", "MACD", "MACD_signal", "MACD_histogram", "bollinger_position", "bollinger_width"])
        
        if self.config.include_volume:
            features.extend(["volume_ratio", "volume_shock", "volume_trend"])
        
        if self.config.include_gaps:
            features.extend(["overnight_gap", "gap_direction"])
        
        features.extend([
            "momentum_5d", "momentum_20d", "momentum_ratio",
            "relative_strength", "outperformance",
        ])
        
        if self.config.include_interactions:
            features.extend([
                "return_x_yield", "rsi_x_gold", "vol_x_vix",
                "momentum_x_yield", "volume_x_risk"
            ])
        
        if include_target:
            features.append("target_rank")
        
        return features

    def drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN features (burn-in period)."""
        feature_cols = self.get_feature_list(include_target=False)
        valid_cols = [c for c in feature_cols if c in df.columns]
        
        df = df.dropna(subset=valid_cols + ["target_rank"])
        logger.info(f"After dropping NaNs: {len(df)} rows")
        
        return df

    def normalize_features(self, df: pd.DataFrame, fit_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features to [-1, 1] range (robust scaling).
        
        Args:
            df: Data to normalize
            fit_data: Data to compute statistics from (if None, use df)
        
        Returns:
            Normalized DataFrame, normalization parameters
        """
        from sklearn.preprocessing import RobustScaler
        
        fit_df = fit_data if fit_data is not None else df
        feature_cols = self.get_feature_list(include_target=False)
        valid_cols = [c for c in feature_cols if c in df.columns]
        
        scaler = RobustScaler()
        
        # Fit on historical data
        scaler.fit(fit_df[valid_cols].fillna(0))
        
        # Transform current data
        df_norm = df.copy()
        df_norm[valid_cols] = scaler.transform(df_norm[valid_cols].fillna(0))
        
        params = {
            "scaler": scaler,
            "feature_cols": valid_cols,
        }
        
        return df_norm, params
