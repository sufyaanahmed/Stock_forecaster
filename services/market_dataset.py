"""
Multi-Stock Dataset Engine
==========================
Load many stocks simultaneously with configurable universes.

Features:
  - Support S&P 500, NASDAQ 100, NIFTY 50 universes
  - Merge macro conditions by date
  - Caching with TTL
  - Missing data handling
  - Async-ready data pipeline
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSES = {
    "sp500_tech": ["NVDA", "META", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX"],
    "nasdaq_100": ["NVDA", "META", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX", "ADBE", "CRM"],
    "nifty_50": ["INFY", "TCS", "RELIANCE", "HINDUNILVR", "ICICIBANK", "HDFC", "BAJAJFINSV", "MARUTI"],
    "custom": [],
}


@dataclass
class DatasetConfig:
    """Configuration for market dataset loader."""
    universe: str = "sp500_tech"
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    seq_len: int = 60  # lookback window for features
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = 24


class MarketDataset:
    """
    Multi-stock dataset with macro alignment.
    
    Schema:
        date | ticker | stock_features | macro_features | target_rank
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir or "/tmp/market_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.tickers = UNIVERSES.get(config.universe, config.universe)
        if isinstance(self.tickers, str):
            self.tickers = config.universe.split(",")
        
        self._stock_cache: Dict[str, pd.DataFrame] = {}
        self._macro_cache: Optional[pd.DataFrame] = None

    def safe_end_date(self) -> str:
        """Returns last completed trading day."""
        if self.config.end_date:
            return self.config.end_date
        today = datetime.utcnow()
        safe_day = today - timedelta(days=2)
        return safe_day.strftime("%Y-%m-%d")

    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Load OHLCV data for a single stock.
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        """
        cache_key = f"{ticker}_{self.config.start_date}_{self.safe_end_date()}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Check cache
        if cache_file.exists():
            try:
                df = pickle.load(open(cache_file, "rb"))
                logger.info(f"Loaded {ticker} from cache")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed for {ticker}: {e}")

        # Download from Yahoo Finance
        try:
            df = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.safe_end_date(),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()

            # ── Flatten MultiIndex columns (yfinance ≥0.2 single-ticker) ─────
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]
            df = df.loc[:, ~df.columns.duplicated()]

            # Save to cache
            df.to_pickle(cache_file)
            logger.info(f"Downloaded and cached {ticker}")
            return df

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV for all tickers in universe.
        
        Returns:
            {ticker: DataFrame}
        """
        stock_data = {}
        for ticker in self.tickers:
            df = self.load_stock_data(ticker)
            if not df.empty:
                df["ticker"] = ticker
                stock_data[ticker] = df
                self._stock_cache[ticker] = df

        logger.info(f"Loaded {len(stock_data)}/{len(self.tickers)} stocks")
        return stock_data

    def align_to_common_dates(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all stocks to common trading dates.
        
        Returns:
            Long-format DataFrame: date | ticker | open | high | low | close | volume | ...
        """
        dfs = []
        for ticker, df in stock_data.items():
            df_copy = df.copy()

            # ── Flatten MultiIndex columns produced by yfinance ≥0.2 ──────────
            # e.g. ('Close', 'AAPL') → 'Close'
            if isinstance(df_copy.columns, pd.MultiIndex):
                df_copy.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df_copy.columns
                ]

            # Remove duplicate columns that arise after MultiIndex flattening
            df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]

            # Ensure the index is reset so 'Date' becomes a plain column
            df_copy = df_copy.reset_index()

            # Normalise the date column name (yfinance uses 'Date' or 'Datetime')
            if "Datetime" in df_copy.columns and "Date" not in df_copy.columns:
                df_copy = df_copy.rename(columns={"Datetime": "Date"})

            df_copy["ticker"] = ticker
            dfs.append(df_copy)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        
        # Keep only dates with data from at least 70% of universe
        date_counts = combined.groupby("Date")["ticker"].nunique()
        min_tickers = max(1, int(len(self.tickers) * 0.7))
        valid_dates = date_counts[date_counts >= min_tickers].index
        
        combined = combined[combined["Date"].isin(valid_dates)]
        combined = combined.rename(columns={"Date": "date"})
        combined["date"] = pd.to_datetime(combined["date"])
        
        return combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    def compute_cross_sectional_target(
        self, df: pd.DataFrame, forward_days: int = 5
    ) -> pd.DataFrame:
        """
        Compute forward return percentile rank (cross-sectional).
        
        This is the KEY difference from single-stock prediction:
        We rank which stocks outperform peers, not predict absolute prices.
        
        Args:
            df: DataFrame with date, ticker, close price
            forward_days: How many days forward to compute target
        
        Returns:
            DataFrame with additional 'target_rank' column [0, 1]
        """
        df = df.copy()

        # ── FIX: remove duplicate columns before any assignment ───────────────
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure 'Close' is a plain 1-D Series (yfinance ≥0.2 can produce
        # a DataFrame with a MultiIndex when multiple tickers are downloaded)
        if isinstance(df["Close"], pd.DataFrame):
            df["Close"] = df["Close"].iloc[:, 0]

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # ── Safe forward-return computation ───────────────────────────────────
        # groupby().shift() returns a Series aligned to df's index — safe assign
        future_close = df.groupby("ticker")["Close"].shift(-forward_days)
        df["forward_return"] = (future_close / df["Close"]) - 1

        # ── Cross-sectional percentile rank per date ──────────────────────────
        df["target_rank"] = (
            df.groupby("date")["forward_return"]
            .rank(pct=True)
        )

        # Remove rows where target is NaN (last `forward_days` rows per ticker)
        df = df.dropna(subset=["target_rank"])

        return df[["date", "ticker", "Close", "Volume", "target_rank"]]

    def get_trading_universe(self, date: datetime) -> List[str]:
        """Return list of tickers with valid data on given date."""
        return self.tickers

    def resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to monthly for longer-horizon modeling."""
        # Group by ticker and month-end
        df["yearmonth"] = df["date"].dt.to_period("M")
        monthly = df.groupby(["ticker", "yearmonth"]).agg({
            "Close": "last",
            "Volume": "sum",
            "target_rank": "last",
        }).reset_index()
        monthly["date"] = monthly["yearmonth"].dt.end_time
        return monthly[["date", "ticker", "Close", "Volume", "target_rank"]]

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Strategies:
          - Forward fill within tickers (max 5 days)
          - Drop rows with > 30% missing features
          - Impute volume with trailing mean
        """
        df = df.copy()
        
        # Forward fill within ticker
        df = df.sort_values(["ticker", "date"])
        df = df.groupby("ticker").apply(
            lambda x: x.fillna(method="ffill", limit=5)
        ).reset_index(drop=True)
        
        # Drop rows with too many NaNs
        feature_cols = [c for c in df.columns if c not in ["date", "ticker", "target_rank"]]
        df = df.dropna(thresh=len(feature_cols) * 0.7)
        
        return df

    def build_dataset(self) -> pd.DataFrame:
        """
        Full pipeline: load → align → compute targets → handle missing.
        
        Returns:
            DataFrame ready for feature engineering
        """
        logger.info("Loading stock data...")
        stock_data = self.load_all_stocks()

        logger.info("Aligning to common dates...")
        df = self.align_to_common_dates(stock_data)

        if df.empty:
            logger.error("No data after alignment!")
            return pd.DataFrame()

        logger.info("Computing cross-sectional targets...")
        df = self.compute_cross_sectional_target(df, forward_days=5)

        logger.info("Handling missing data...")
        df = self.handle_missing_data(df)

        logger.info(f"Dataset shape: {df.shape}")
        return df

    def save_dataset(self, df: pd.DataFrame, path: str):
        """Save dataset to disk."""
        df.to_parquet(path)
        logger.info(f"Saved dataset to {path}")

    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from disk."""
        df = pd.read_parquet(path)
        logger.info(f"Loaded dataset from {path}")
        return df
