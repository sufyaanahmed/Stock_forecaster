"""
Data Provider Abstraction Layer
================================
Pluggable architecture: swap data sources without touching the feature pipeline.

Supported free providers:
  1. yfinance      — Yahoo Finance (default). Stocks, ETFs, crypto, forex.
  2. stooq         — Polish/global historical data via pandas-datareader.
                     Good fallback for equities when yfinance rate-limits.
  3. alpha_vantage — Generous free tier (25 req/day standard, 500/day premium).
                     Good for US equities + FX + crypto.
  4. finnhub       — Free tier: 60 calls/min. Real-time quotes, company info.
  5. fred          — Federal Reserve Economic Data. Macroeconomic indicators
                     (interest rates, CPI, unemployment, bond yields, DXY).
                     Completely free, no auth needed for most series.

Usage:
    from data.providers import get_provider, fetch_ohlcv

    df = fetch_ohlcv("AAPL", start="2015-01-01")   # uses default provider
    df = fetch_ohlcv("AAPL", provider="stooq")      # explicit override
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

# ── Disk-level cache (avoids network calls during development) ─────────────────
# Files stored in a local .cache/ directory.  Set QUANTML_NO_CACHE=1 to disable.
import hashlib, pickle
from pathlib import Path

CACHE_DIR  = Path(__file__).parent.parent / ".cache" / "market_data"
CACHE_TTL  = 4 * 3600   # 4 hours (in seconds)
NO_CACHE   = os.environ.get("QUANTML_NO_CACHE", "0") == "1"


def _disk_get(key: str) -> Optional[pd.DataFrame]:
    if NO_CACHE:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > CACHE_TTL:
            path.unlink()
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _disk_set(key: str, df: pd.DataFrame) -> None:
    if NO_CACHE:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def _cache_key(provider: str, ticker: str, start: str, end: str) -> str:
    raw = f"{provider}:{ticker}:{start}:{end}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Base Provider ─────────────────────────────────────────────────────────────

class DataProvider(ABC):
    """All providers must return a DataFrame with lowercase OHLCV columns
    and a DatetimeIndex.  Minimum required columns: open, high, low, close, volume."""

    name: str = "base"

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        ...

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df = df.copy()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        df         = df.sort_index().dropna(how="all")
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = float("nan")
        return df[["open", "high", "low", "close", "volume"]]


# ── Providers ─────────────────────────────────────────────────────────────────

class YFinanceProvider(DataProvider):
    """
    Yahoo Finance via the yfinance library.
    - Completely free, no API key required.
    - Covers: US/global equities, ETFs, indices, crypto (BTC-USD), FX (EURUSD=X).
    - Rate limit: ~2,000 req/hour informally; no published limit.
    - Weakness: undocumented API, occasionally breaks on yfinance version changes.
    """
    name = "yfinance"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
        if df is None or df.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        return self._normalize(df)


class StooqProvider(DataProvider):
    """
    Stooq historical data via pandas-datareader.
    - Free, no key required.
    - Good fallback for US equities and Polish stocks.
    - Tickers: '^GSPC' for S&P500, 'AAPL.US' for Apple.
    - Install: pip install pandas-datareader
    """
    name = "stooq"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            raise ImportError("Install pandas-datareader: pip install pandas-datareader")
        # Stooq uses '.US' suffix for US stocks
        stooq_ticker = ticker if "." in ticker else f"{ticker}.US"
        df = pdr.DataReader(stooq_ticker, "stooq", start=start, end=end)
        if df is None or df.empty:
            raise ValueError(f"stooq returned no data for {ticker}")
        return self._normalize(df)


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage — generous free tier (25 req/day without key, 500/day with free key).
    - Get a free key at: https://www.alphavantage.co/support/#api-key
    - Set env var: ALPHA_VANTAGE_KEY=your_key
    - Covers: US equities, FX, crypto, commodities, technical indicators.
    - Install: pip install alpha_vantage
    """
    name = "alpha_vantage"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        key = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
        try:
            from alpha_vantage.timeseries import TimeSeries
        except ImportError:
            raise ImportError("Install alpha_vantage: pip install alpha_vantage")
        ts = TimeSeries(key=key, output_format="pandas")
        df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        df.columns = ["open","high","low","close","adj_close","volume","div","split"]
        df = df[["open","high","low","close","volume"]]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return self._normalize(df.loc[start:end])


class FredProvider(DataProvider):
    """
    FRED (Federal Reserve Economic Data) — completely free, no key needed.
    - Best for: interest rates, CPI, GDP, unemployment, bond yields, DXY.
    - FRED series IDs (examples):
        'DGS10'    : 10-Year Treasury Yield
        'FEDFUNDS' : Federal Funds Rate
        'CPIAUCSL' : CPI All Items
        'DTWEXBGS' : DXY (Dollar Index)
        'VIXCLS'   : CBOE VIX
        'DCOILWTICO': WTI Crude Oil price
        'GOLDAMGBD228NLBM': Gold price (London PM fix)
    - These are single-series, not OHLCV; the 'close' column holds the value.
    - Install: pip install pandas-datareader
    """
    name = "fred"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            raise ImportError("Install pandas-datareader: pip install pandas-datareader")
        df = pdr.DataReader(ticker, "fred", start=start, end=end)
        if df is None or df.empty:
            raise ValueError(f"FRED returned no data for series {ticker}")
        df = df.rename(columns={df.columns[0]: "close"})
        for col in ["open","high","low","volume"]:
            df[col] = df["close"]
        return self._normalize(df[["open","high","low","close","volume"]])


class FinnhubProvider(DataProvider):
    """
    Finnhub — free tier: 60 API calls/minute.
    - Set env var: FINNHUB_KEY=your_key (get at https://finnhub.io)
    - Covers: US/global equities, ETFs, crypto, FX, company fundamentals.
    - Install: pip install finnhub-python
    """
    name = "finnhub"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        key = os.environ.get("FINNHUB_KEY", "")
        if not key:
            raise ValueError("Set env var FINNHUB_KEY for Finnhub provider")
        try:
            import finnhub
        except ImportError:
            raise ImportError("Install finnhub-python: pip install finnhub-python")
        client = finnhub.Client(api_key=key)
        t_start = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        t_end   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp())
        res     = client.stock_candles(ticker, "D", t_start, t_end)
        if res.get("s") != "ok":
            raise ValueError(f"Finnhub returned status={res.get('s')} for {ticker}")
        df = pd.DataFrame({
            "open":   res["o"], "high": res["h"],
            "low":    res["l"], "close": res["c"],
            "volume": res["v"],
        }, index=pd.to_datetime(res["t"], unit="s"))
        return self._normalize(df)


# ── Registry + public fetch function ──────────────────────────────────────────

_PROVIDERS: Dict[str, DataProvider] = {
    "yfinance":      YFinanceProvider(),
    "stooq":         StooqProvider(),
    "alpha_vantage": AlphaVantageProvider(),
    "fred":          FredProvider(),
    "finnhub":       FinnhubProvider(),
}

# Provider priority: try in order until one succeeds
DEFAULT_CHAIN = ["yfinance", "stooq"]


def get_provider(name: str) -> DataProvider:
    """Return a registered provider by name."""
    p = _PROVIDERS.get(name)
    if p is None:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_PROVIDERS)}")
    return p


def fetch_ohlcv(
    ticker: str,
    start:  str,
    end:    Optional[str] = None,
    provider: Optional[str] = None,
    fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data with disk cache + optional multi-provider fallback.

    Args:
        ticker:   Instrument symbol (e.g. 'AAPL', 'BTC-USD', 'EURUSD=X')
        start:    Start date 'YYYY-MM-DD'
        end:      End date 'YYYY-MM-DD' (defaults to yesterday)
        provider: Force a specific provider (default: try DEFAULT_CHAIN in order)
        fallback: If True and primary provider fails, try remaining chain

    Returns:
        DataFrame with lowercase OHLCV columns and DatetimeIndex
    """
    if end is None:
        end = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

    chain = [provider] if provider else DEFAULT_CHAIN

    key = _cache_key(chain[0], ticker, start, end)
    cached = _disk_get(key)
    if cached is not None:
        return cached

    last_exc = None
    for pname in chain:
        try:
            p  = get_provider(pname)
            df = p.fetch(ticker, start, end)
            if df is not None and not df.empty:
                _disk_set(key, df)
                return df
        except Exception as exc:
            last_exc = exc
            print(f"[providers] {pname} failed for {ticker}: {exc}")
            if not fallback:
                raise

    raise RuntimeError(
        f"All providers failed for {ticker}: {last_exc}"
    )


# ── Macro / intermarket helpers ───────────────────────────────────────────────
# These are ready to use but not yet wired into the training pipeline.
# See TECHNICAL_DOCUMENTATION.md → Section: Macroeconomic Feature Integration

MACRO_SERIES = {
    "10y_yield":    "DGS10",          # 10-Year US Treasury Yield
    "fed_funds":    "FEDFUNDS",       # Federal Funds Rate
    "cpi":          "CPIAUCSL",       # CPI (monthly, All Items)
    "dxy":          "DTWEXBGS",       # US Dollar Index (trade-weighted)
    "vix":          "VIXCLS",         # CBOE VIX Fear Index
    "oil_wti":      "DCOILWTICO",     # WTI Crude Oil spot price
    "gold":         "GOLDAMGBD228NLBM",  # Gold, London PM Fix (USD/troy oz)
    "unemployment": "UNRATE",         # US Unemployment Rate (monthly)
    "gdp":          "GDPC1",          # Real GDP (quarterly)
}

def fetch_macro(series_id: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch a macroeconomic time series from FRED.

    Example:
        vix = fetch_macro("VIXCLS", "2015-01-01")
        gold = fetch_macro("GOLDAMGBD228NLBM", "2015-01-01")
    """
    return fetch_ohlcv(series_id, start=start, end=end, provider="fred", fallback=False)
