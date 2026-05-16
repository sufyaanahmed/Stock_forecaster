"""
Data Pipeline — Leak-Free Financial Feature Engineering
=======================================================
Key principles:
  1. No look-ahead bias
  2. Returns instead of prices (stationarity)
  3. Rolling stats strictly past-only
  4. Robust Yahoo Finance handling
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    ticker: str
    start: str = "2008-01-01"
    end: Optional[str] = None
    seq_len: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15


# ─────────────────────────────────────────────────────────────
# SAFE DATE HANDLING (IMPORTANT FIX)
# ─────────────────────────────────────────────────────────────
def safe_end_date():
    """
    Prevents Yahoo Finance edge-case failures.
    Always uses last fully completed trading day.
    """
    today = datetime.utcnow()
    safe_day = today - timedelta(days=2)
    return safe_day.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
# FETCH DATA (FIXED)
# ─────────────────────────────────────────────────────────────
def fetch_raw(cfg: DataConfig) -> pd.DataFrame:

    end_date = cfg.end if cfg.end else safe_end_date()

    print(f"Downloading {cfg.ticker} from {cfg.start} to {end_date}")

    df = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {cfg.ticker}")

    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    # FIX: handle MultiIndex safety
    df.columns = [
        c[0].lower() if isinstance(c, tuple) else c.lower()
        for c in df.columns
    ]

    if len(df) < 300:
        raise ValueError(f"Insufficient data: {len(df)} rows")

    return df


# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:

    feat = pd.DataFrame(index=df.index)

    close = df["close"]

    # Returns
    feat["log_return"] = np.log(close / close.shift(1))
    feat["log_return_2"] = np.log(close / close.shift(2))
    feat["log_return_5"] = np.log(close / close.shift(5))
    feat["log_return_10"] = np.log(close / close.shift(10))

    # Volatility
    feat["vol_10"] = feat["log_return"].rolling(10, min_periods=5).std()
    feat["vol_20"] = feat["log_return"].rolling(20, min_periods=10).std()
    feat["vol_60"] = feat["log_return"].rolling(60, min_periods=30).std()
    feat["vol_ratio"] = feat["vol_10"] / (feat["vol_60"] + 1e-8)

    # RSI
    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_28"] = _rsi(close, 28)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_norm"] = (macd - signal) / (close + 1e-8)

    # Bollinger
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    feat["bb_position"] = (close - sma) / (2 * std + 1e-8)
    feat["bb_width"] = (2 * std) / (sma + 1e-8)

    # Volume
    feat["volume_ratio"] = np.log(
        df["volume"] / (df["volume"].rolling(20, min_periods=5).mean() + 1e-8)
    )

    # High-low range
    feat["hl_ratio"] = np.log(df["high"] / (df["low"] + 1e-8))

    # TARGET (next-day return)
    feat["target"] = feat["log_return"].shift(-1)

    return feat.dropna()


# ─────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────
def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────
# STATIONARITY CHECK
# ─────────────────────────────────────────────────────────────
def check_stationarity(series: pd.Series):

    result = adfuller(series.dropna())

    p = result[1]

    print("\nADF Test:")
    print(f"p-value: {p:.5f}")
    print("Stationary [OK]" if p < 0.05 else "Non-stationary [WARN]")

    return p < 0.05


# ─────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────
def split_data(feat: pd.DataFrame, cfg: DataConfig):

    n = len(feat)

    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train = feat.iloc[:train_end]
    val = feat.iloc[train_end:val_end]
    test = feat.iloc[val_end:]

    return train, val, test


# ─────────────────────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────────────────────
def scale_features(train, val, test, target="target"):

    features = [c for c in train.columns if c != target]

    scaler = RobustScaler()

    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])

    return (
        X_train,
        X_val,
        X_test,
        train[target].values,
        val[target].values,
        test[target].values,
        scaler,
        features,   # ← now returned so pipeline can surface it
    )


# ─────────────────────────────────────────────────────────────
# SEQUENCES
# ─────────────────────────────────────────────────────────────
def make_sequences(X, y, seq_len):

    Xs, ys = [], []

    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])

    return np.array(Xs), np.array(ys)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def load_and_prepare(cfg: DataConfig):

    print(f"\nLoading {cfg.ticker}...")

    raw = fetch_raw(cfg)
    feat = compute_features(raw)

    if not check_stationarity(feat["target"]):
        raise ValueError("Target not stationary")

    train, val, test = split_data(feat, cfg)

    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler, feature_cols = scale_features(
        train, val, test
    )

    X_tr, y_tr = make_sequences(X_tr, y_tr, cfg.seq_len)
    X_val, y_val = make_sequences(X_val, y_val, cfg.seq_len)
    X_te, y_te = make_sequences(X_te, y_te, cfg.seq_len)

    print("\nShapes:")
    print(X_tr.shape, X_val.shape, X_te.shape)

    return {
        "X_train":      X_tr,
        "y_train":      y_tr,
        "X_val":        X_val,
        "y_val":        y_val,
        "X_test":       X_te,
        "y_test":       y_te,
        "scaler":       scaler,
        "feature_cols": feature_cols,   # ← always present now
        "raw":          raw,
        "feat":         feat,
    }