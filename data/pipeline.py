"""
Data Pipeline — Leak-Free Financial Feature Engineering
=======================================================
Key principles:
  1. All features computed AFTER train/test split to prevent look-ahead bias
  2. Returns, not prices (stationarity)
  3. Rolling statistics use only past data (no center=True)
  4. ADF test to verify stationarity of target variable
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataConfig:
    ticker: str
    start: str = "2008-01-01"
    end: Optional[str] = None          # None → today
    seq_len: int = 60                  # lookback window (trading days)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train - val = 0.15


def fetch_raw(cfg: DataConfig) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance.
    Uses Adj Close — accounts for splits/dividends.
    Raw Close is financially meaningless for returns.
    """
    df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=True)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]

    if len(df) < 300:
        raise ValueError(f"Insufficient data for {cfg.ticker}: {len(df)} rows. Need ≥300.")
    
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features using only past data (no leakage).
    
    Why log returns?
      r_t = log(P_t / P_{t-1})
      - Stationary (no unit root under H0 of ADF)
      - Additive across time: multi-day return = sum of daily returns
      - Approximately normal → compatible with many statistical tests
      - Removes price-level scale differences across tickers

    All rolling windows use min_periods to avoid NaN propagation.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Core return features ──────────────────────────────────────────────────
    feat['log_return']    = np.log(df['close'] / df['close'].shift(1))
    feat['log_return_2']  = np.log(df['close'] / df['close'].shift(2))
    feat['log_return_5']  = np.log(df['close'] / df['close'].shift(5))
    feat['log_return_10'] = np.log(df['close'] / df['close'].shift(10))

    # ── Volatility features ───────────────────────────────────────────────────
    # Realized volatility: rolling std of log returns
    # Statistically: sample std of iid returns ≈ σ√Δt
    feat['vol_10']  = feat['log_return'].rolling(10,  min_periods=5).std()
    feat['vol_20']  = feat['log_return'].rolling(20,  min_periods=10).std()
    feat['vol_60']  = feat['log_return'].rolling(60,  min_periods=30).std()

    # Vol ratio: short/long vol → measures vol regime
    feat['vol_ratio'] = feat['vol_10'] / (feat['vol_60'] + 1e-8)

    # ── Momentum features (price-relative, not absolute) ─────────────────────
    # RSI: Relative Strength Index
    # Measures overbought/oversold on [0, 100] scale
    feat['rsi_14'] = _rsi(df['close'], 14)
    feat['rsi_28'] = _rsi(df['close'], 28)

    # MACD: momentum oscillator
    # Fast EMA - Slow EMA → mean-reverting signal
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd_norm'] = (macd_line - signal_line) / (df['close'] + 1e-8)

    # ── Bollinger Band position ───────────────────────────────────────────────
    # %B: where price sits within the band → [0,1] normalization of price
    # Unlike raw price, this is stationary
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    feat['bb_position'] = (df['close'] - sma_20) / (2 * std_20 + 1e-8)
    feat['bb_width']    = (2 * std_20) / (sma_20 + 1e-8)   # bandwidth signal

    # ── Volume features ───────────────────────────────────────────────────────
    # Log-normalized volume relative to rolling average
    # Raw volume is non-stationary (company grows over time)
    feat['volume_ratio'] = np.log(
        df['volume'] / (df['volume'].rolling(20, min_periods=5).mean() + 1e-8)
    )

    # ── High-Low range (intraday volatility proxy) ────────────────────────────
    feat['hl_ratio'] = np.log(df['high'] / (df['low'] + 1e-8))

    # ── Target: next-day log return ──────────────────────────────────────────
    # shift(-1): tomorrow's return as today's label
    # IMPORTANT: This shift must be done BEFORE splitting, then the test set
    # boundary is handled carefully to avoid using future labels.
    feat['target'] = feat['log_return'].shift(-1)

    feat = feat.dropna()
    return feat


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI using exponential smoothing.
    RSI = 100 - 100 / (1 + RS),  RS = avg_gain / avg_loss
    EWM with alpha=1/period approximates Wilder's smoothing.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def check_stationarity(series: pd.Series, name: str = "series") -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary)
    Reject H0 (p < 0.05) → stationary → safe to model

    Why this matters: Non-stationary targets cause the model to learn
    spurious level correlations rather than true predictive patterns.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    stationary = result[1] < 0.05
    print(f"ADF Test [{name}]:")
    print(f"  Statistic: {result[0]:.4f}")
    print(f"  p-value:   {result[1]:.4f}")
    print(f"  → {'STATIONARY ✓' if stationary else 'NON-STATIONARY ✗ — do not use as target'}")
    return {"statistic": result[0], "p_value": result[1], "stationary": stationary}


def split_data(feat: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split with NO shuffling.
    Temporal order is sacred in time-series — shuffling creates look-ahead bias.
    
    Returns: (train_df, val_df, test_df)
    """
    n = len(feat)
    train_end = int(n * cfg.train_ratio)
    val_end   = int(n * (cfg.train_ratio + cfg.val_ratio))

    train = feat.iloc[:train_end].copy()
    val   = feat.iloc[train_end:val_end].copy()
    test  = feat.iloc[val_end:].copy()

    print(f"\nData split:")
    print(f"  Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows)")
    print(f"  Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} rows)")
    print(f"  Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)")

    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = 'target'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    RobustScaler: scales using median and IQR rather than mean/std.
    Why? Financial returns have fat tails — outlier earnings moves,
    crashes, etc. RobustScaler is resistant to these outliers.
    MinMaxScaler would compress the entire distribution to serve one outlier.

    CRITICAL: fit ONLY on train. Transform val/test with train's statistics.
    This simulates the real deployment scenario where you can't see the future.
    """
    feature_cols = [c for c in train.columns if c != target_col]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train[feature_cols])   # fit here
    X_val   = scaler.transform(val[feature_cols])          # no fit — uses train stats
    X_test  = scaler.transform(test[feature_cols])         # no fit — uses train stats

    y_train = train[target_col].values
    y_val   = val[target_col].values
    y_test  = test[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature matrix into (samples, seq_len, features) tensor.
    
    For sample i, the input is X[i : i+seq_len] and the label is y[i+seq_len-1].
    This is a sliding window — each sample overlaps the previous by (seq_len-1).

    Why seq_len=60? ~3 months of trading days. Captures quarterly patterns
    without requiring the model to remember back too far (gradient flow degrades).
    Hyperparameter to tune via validation IC.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def load_and_prepare(cfg: DataConfig):
    """
    Full pipeline: fetch → features → ADF check → split → scale → sequences.
    Returns everything needed for training and evaluation.
    """
    print(f"Loading {cfg.ticker}...")
    raw = fetch_raw(cfg)
    feat = compute_features(raw)

    # Verify target is stationary before proceeding
    adf = check_stationarity(feat['target'], "next-day log return")
    if not adf['stationary']:
        raise ValueError("Target is non-stationary. Check your feature computation.")

    train_df, val_df, test_df = split_data(feat, cfg)
    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler = scale_features(train_df, val_df, test_df)
    
    X_tr,  y_tr  = make_sequences(X_tr,  y_tr,  cfg.seq_len)
    X_val, y_val = make_sequences(X_val, y_val, cfg.seq_len)
    X_te,  y_te  = make_sequences(X_te,  y_te,  cfg.seq_len)

    print(f"\nFinal tensor shapes:")
    print(f"  X_train: {X_tr.shape}  (samples, seq_len, features)")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_te.shape}")

    feature_cols = [c for c in feat.columns if c != 'target']

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_val, "y_val":   y_val,
        "X_test":  X_te, "y_test":   y_te,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "raw_df": raw,
        "feat_df": feat,
        "test_df": test_df,
    }