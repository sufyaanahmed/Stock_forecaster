"""
Macro Data Layer
================
Load and process macro-economic data that applies to ALL stocks.

Macro indicators:
  - Gold futures (GC=F or GLD)
  - US 10Y yields (^TNX)
  - Fed funds rate (FRED or alternative)
  - VIX (optional volatility regime)
  - DXY (optional currency)
  - Oil (optional)

These apply by DATE to all stocks in universe.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroConfig:
    """Configuration for macro data loader."""
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    use_gold: bool = True
    use_yields: bool = True
    use_fed_rate: bool = True
    use_vix: bool = True
    use_dxy: bool = False
    use_oil: bool = False
    cache_dir: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# MACRO DATA SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class MacroService:
    """
    Centralized macro data provider.
    
    Loads once, indexes by date, applies to all stocks.
    This enables macro-regime awareness in the model.
    """

    SYMBOLS = {
        "gold": "GLD",  # Gold ETF (alternative: GC=F)
        "yields": "^TNX",  # US 10Y Treasury Yield
        "vix": "^VIX",  # Volatility Index
        "dxy": "^DXY",  # Dollar Index
        "oil": "CL=F",  # Oil Futures
    }

    def __init__(self, config: MacroConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir or "/tmp/macro_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._macro_df: Optional[pd.DataFrame] = None

    def safe_end_date(self) -> str:
        """Returns last completed trading day."""
        if self.config.end_date:
            return self.config.end_date
        today = datetime.utcnow()
        safe_day = today - timedelta(days=2)
        return safe_day.strftime("%Y-%m-%d")

    def load_macro_series(self, symbol: str, name: str) -> Optional[pd.Series]:
        """
        Load single macro indicator from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., "^TNX", "GLD")
            name: Name for the series
        
        Returns:
            Series indexed by date
        """
        cache_file = self.cache_dir / f"{name}_{self.config.start_date}_{self.safe_end_date()}.pkl"
        
        # Check cache
        if cache_file.exists():
            try:
                series = pickle.load(open(cache_file, "rb"))
                if isinstance(series, pd.Series):
                    logger.info(f"Loaded {name} from cache")
                    return series
                logger.warning(f"Cached macro data for {name} is invalid, redownloading")
            except Exception as e:
                logger.warning(f"Cache read failed for {name}: {e}")

        try:
            df = yf.download(
                symbol,
                start=self.config.start_date,
                end=self.safe_end_date(),
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                if ("Adj Close", symbol) in df.columns:
                    series = df[("Adj Close", symbol)]
                elif ("Close", symbol) in df.columns:
                    series = df[("Close", symbol)]
                elif "Adj Close" in df.columns.get_level_values(0):
                    col = [c for c in df.columns if c[0] == "Adj Close"][0]
                    series = df[col]
                elif "Close" in df.columns.get_level_values(0):
                    col = [c for c in df.columns if c[0] == "Close"][0]
                    series = df[col]
                else:
                    logger.warning(f"Macro symbol {symbol} missing Close/Adj Close columns")
                    return None
            else:
                if "Adj Close" in df.columns:
                    series = df["Adj Close"]
                elif "Close" in df.columns:
                    series = df["Close"]
                else:
                    logger.warning(f"Macro symbol {symbol} missing Close/Adj Close columns")
                    return None

            series = pd.Series(series).copy()
            series.name = name
            series.to_pickle(cache_file)
            logger.info(f"Downloaded and cached {name}")
            return series

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return None

    def load_all_macro(self) -> pd.DataFrame:
        """
        Load all configured macro indicators.
        
        Returns:
            DataFrame with columns: gold, yields, vix, dxy, oil, ...
            Indexed by date
        """
        macro_data = {}

        if self.config.use_gold:
            gold = self.load_macro_series(self.SYMBOLS["gold"], "gold")
            if gold is not None:
                macro_data["gold"] = gold

        if self.config.use_yields:
            yields = self.load_macro_series(self.SYMBOLS["yields"], "yields")
            if yields is not None:
                macro_data["yields"] = yields

        if self.config.use_vix:
            vix = self.load_macro_series(self.SYMBOLS["vix"], "vix")
            if vix is not None:
                macro_data["vix"] = vix

        if self.config.use_dxy:
            dxy = self.load_macro_series(self.SYMBOLS["dxy"], "dxy")
            if dxy is not None:
                macro_data["dxy"] = dxy

        if self.config.use_oil:
            oil = self.load_macro_series(self.SYMBOLS["oil"], "oil")
            if oil is not None:
                macro_data["oil"] = oil

        # Keep only valid series values
        valid_macro_data = {
            name: series
            for name, series in macro_data.items()
            if isinstance(series, pd.Series)
        }

        if not valid_macro_data:
            logger.warning("No macro data loaded!")
            return pd.DataFrame()

        # Combine into DataFrame
        macro_df = pd.concat(valid_macro_data, axis=1, keys=valid_macro_data.keys())
        macro_df.index.name = "date"
        macro_df = macro_df.sort_index()

        logger.info(f"Loaded macro data with {len(macro_df)} dates")
        return macro_df

    def compute_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered macro features from raw prices.
        
        Features:
          - gold_return_1d, gold_return_5d
          - gold_volatility_10
          - yield_change_1d, yield_change_5d
          - yield_momentum
          - vix_level, vix_change
          - dxy_return
          - oil_return
        """
        df = df.copy()

        # Gold features
        if "gold" in df.columns:
            df["gold_return_1d"] = df["gold"].pct_change(1)
            df["gold_return_5d"] = df["gold"].pct_change(5)
            df["gold_volatility_10"] = df["gold_return_1d"].rolling(10).std()

        # Yields features
        if "yields" in df.columns:
            df["yield_change_1d"] = df["yields"].diff(1)
            df["yield_change_5d"] = df["yields"].diff(5)
            df["yield_momentum"] = df["yields"].rolling(10).mean() - df["yields"].rolling(60).mean()

        # VIX features
        if "vix" in df.columns:
            df["vix_level"] = df["vix"]
            df["vix_change"] = df["vix"].pct_change(1)
            df["vix_high_regime"] = (df["vix"] > df["vix"].rolling(60).mean()).astype(float)

        # DXY features
        if "dxy" in df.columns:
            df["dxy_return"] = df["dxy"].pct_change(1)

        # Oil features
        if "oil" in df.columns:
            df["oil_return"] = df["oil"].pct_change(1)

        # Risk regime indicator
        # Risk-on: yields low, VIX low, gold weak, growth strong
        # Risk-off: yields high, VIX high, gold strong, growth weak
        if "yields" in df.columns and "vix" in df.columns:
            yields_percentile = df["yields"].rolling(60).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
            vix_percentile = df["vix"].rolling(60).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
            df["risk_regime_score"] = yields_percentile * 0.5 + vix_percentile * 0.5
            df["risk_regime"] = df["risk_regime_score"].apply(
                lambda x: "risk_off" if x > 0.6 else ("risk_on" if x < 0.4 else "neutral")
            )

        return df

    def get_macro_snapshot(self, date: datetime) -> Dict[str, float]:
        """
        Get macro features for a specific date.
        
        Returns:
            Dict of macro features
        """
        if self._macro_df is None:
            self._macro_df = self.load_all_macro()
            self._macro_df = self.compute_macro_features(self._macro_df)

        if self._macro_df.empty:
            logger.warning("Macro snapshot requested but no macro data is available")
            return {}

        date_str = pd.Timestamp(date).normalize()
        if date_str not in self._macro_df.index:
            logger.warning(f"No macro data for {date_str}, using nearest")
            # Find nearest date
            idx = self._macro_df.index.get_indexer([date_str], method="nearest")
            if idx[0] >= 0:
                date_str = self._macro_df.index[idx[0]]
            else:
                return {}

        row = self._macro_df.loc[date_str]
        return row.to_dict()

    def get_macro_context(self, date: datetime) -> Dict[str, str]:
        """
        Interpretable macro context for a date.
        
        Returns:
            {gold_trend, yield_trend, risk_regime, etc.}
        """
        snapshot = self.get_macro_snapshot(date)
        
        context = {
            "date": str(date),
            "gold_trend": "rising" if snapshot.get("gold_return_5d", 0) > 0 else "falling",
            "yield_trend": "rising" if snapshot.get("yield_change_5d", 0) > 0 else "falling",
            "gold_volatility": "high" if snapshot.get("gold_volatility_10", 0) > 0.02 else "normal",
            "vix_level": "elevated" if snapshot.get("vix_level", 0) > 20 else "calm",
            "risk_regime": snapshot.get("risk_regime", "neutral"),
        }
        
        return context

    def align_to_dates(self, dates: List[datetime]) -> pd.DataFrame:
        """
        Return macro features aligned to specific dates.
        
        Args:
            dates: List of dates
        
        Returns:
            DataFrame with macro features, indexed by date
        """
        if self._macro_df is None:
            self._macro_df = self.load_all_macro()
            self._macro_df = self.compute_macro_features(self._macro_df)

        if self._macro_df.empty:
            # Return an empty DataFrame with requested index if no macro data exists
            dates = pd.to_datetime(dates)
            return pd.DataFrame(index=dates)

        # Convert dates to timestamps
        dates = pd.to_datetime(dates)
        
        # Reindex to requested dates (forward fill if needed)
        macro_subset = self._macro_df.reindex(dates, method="ffill")
        
        return macro_subset

    def save_macro_cache(self, path: str):
        """Save computed macro data to disk."""
        if self._macro_df is not None:
            self._macro_df.to_parquet(path)
            logger.info(f"Saved macro cache to {path}")

    def load_macro_cache(self, path: str):
        """Load pre-computed macro data from disk."""
        self._macro_df = pd.read_parquet(path)
        logger.info(f"Loaded macro cache from {path}")
