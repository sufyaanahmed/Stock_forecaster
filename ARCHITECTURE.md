"""
ARCHITECTURAL UPGRADE DOCUMENTATION
====================================

This document explains the major architectural upgrade from single-stock 
LSTM forecasting to macro-aware multi-stock ranking.

## What Changed

### Before (Single-Stock LSTM)
- Input: Historical OHLCV for one stock
- Output: Tomorrow's return for that stock (+0.5% or -0.3%)
- Assumption: Stocks move independently
- Model: 2-layer LSTM with attention
- Problem: Isolation assumption is wrong; macro environment matters

### After (Multi-Stock Macro-Aware Ranking)
- Input: All stocks' features + macro conditions + date
- Output: Ranking scores for all stocks relative to each other
- Assumption: Stocks move differently under different macro regimes
- Model: LightGBM LambdaRank (cross-sectional ranking)
- Solution: Learn "which stocks outperform peers?"

## Why This Matters

### Single-Stock Prediction Limitations
1. Ignores market structure (dispersion of returns changes over time)
2. No macro regime awareness (growth stocks outperform in risk-on, underperform in risk-off)
3. Doesn't capture relative value (NVDA vs META relative valuation)
4. Limited alpha (all single-stock models converge to similar signals)

### Multi-Stock Ranking Advantages
1. **Market Structure**: Learns how stocks disperse relative to each other
2. **Macro Regimes**: Incorporates gold, yields, VIX, fed rates
3. **Relative Value**: Ranks NVDA vs META vs AAPL, not just "NVDA up/down"
4. **Portfolio Construction**: Directly outputs long/short candidates
5. **Risk Management**: Can filter by regime (reduce leverage in risk-off)

## Architecture Overview

### Directory Structure

```
stock-forecaster/
├── services/
│   ├── market_dataset.py          # Multi-stock data alignment
│   ├── macro_service.py           # Gold, yields, VIX loading
│   ├── feature_engineering.py     # 40+ technical + macro features
│   ├── ranking_service.py         # End-to-end ranking pipeline
│   ├── backtest_service.py        # Portfolio backtesting
│   └── (existing: market_dataset.py, etc.)
│
├── models/
│   ├── ranker.py                  # LightGBM LambdaRank model
│   ├── model_registry.py          # Model switching (legacy vs market)
│   └── (existing: lstm.py, train.py)
│
├── routes/
│   ├── analyze.py                 # NEW: Market analysis endpoints
│   ├── legacy_routes.py           # Keep old single-stock endpoints
│   └── __init__.py
│
├── config/
│   ├── market_config.py           # Central configuration
│   └── __init__.py
│
├── api/
│   └── main.py                    # FastAPI integration point
│
├── run_market_train.py            # Training script for ranker
├── requirements.txt               # Add lightgbm
└── Readme.md                      # Updated documentation
```

## Component Details

### 1. Market Dataset (market_dataset.py)

**Purpose**: Load and align multiple stocks to common trading dates

```python
universe = ["NVDA", "META", "AAPL", "MSFT", "TSLA", ...]

# Load all stocks for date range
dataset.load_all_stocks()

# Align to common dates (≥70% of universe has data)
df = dataset.align_to_common_dates(stock_data)

# Schema: date | ticker | open | high | low | close | volume | target_rank
```

**Key Features**:
- Configurable universes (S&P 500, NASDAQ 100, etc.)
- Data caching (avoid re-downloading)
- Missing data handling (forward fill, drop incomplete dates)
- **Cross-sectional target**: Future return percentile rank (0-1)

**Target Generation**:
```python
# For each date, rank forward returns
date | ticker | return_tomorrow | rank_tomorrow
2026-05-18 | NVDA | +2.5%        | 0.95  (top performer)
2026-05-18 | AAPL | -0.5%        | 0.15
2026-05-18 | META | +1.2%        | 0.72
```

### 2. Macro Service (macro_service.py)

**Purpose**: Load macro indicators and compute macro features

**Indicators**:
- Gold futures (GLD)
- US 10Y yields (^TNX)
- VIX (^VIX)
- Fed funds rate (optional)
- DXY, Oil (optional)

**Computed Features**:
```python
gold_return_1d, gold_return_5d, gold_volatility_10
yield_change_1d, yield_change_5d, yield_momentum
vix_level, vix_change, vix_high_regime
risk_regime_score: "risk_on" | "neutral" | "risk_off"
```

**Key Design**: 
- Macro features apply to **all stocks on a date**
- Indexed by date (easy to merge with stock features)
- Regime detection for portfolio filtering

### 3. Feature Engineering (feature_engineering.py)

**Purpose**: Generate 40+ stock and macro features for ranking

**Stock Features** (~25):
- Returns: 1d, 5d, 20d
- Volatility: 10-day, 60-day, ratio
- Technical: RSI, MACD, Bollinger position
- Volume: ratio, shock, trend
- Gaps: overnight gaps
- Momentum: 5d, 20d, ratio
- Relative strength: vs universe

**Macro Features** (~8):
- Gold returns & volatility
- Yield changes & momentum
- VIX levels & regime
- Fed rate effects

**Interaction Features** (~10):
```python
return_1d * yield_change        # Growth sensitivity to rates
RSI * gold_momentum             # Mean reversion under gold stress
volatility * vix_level          # Vol clustering
momentum * yield_momentum       # Momentum drift
volume * risk_regime_score      # Volume under stress
```

**Key Design**:
- Engineered for **relative ranking**, not price prediction
- Normalized with RobustScaler (handles outliers)
- NaN handling: drop incomplete rows after feature burn-in period

### 4. Ranker Model (ranker.py)

**Purpose**: LightGBM LambdaRank for cross-sectional ranking

**Parameters**:
```python
objective: "lambdarank"
metric: "ndcg"  # Normalized Discounted Cumulative Gain
num_leaves: 31
learning_rate: 0.1
num_boost_rounds: 100 (default)
```

**Data Format** (LambdaRank-specific):
```python
X: (n_samples, n_features)       # All stock-date features
y: (n_samples,)                  # Percentile ranks [0, 1]
group_sizes: (n_dates,)          # Stocks per date
```

**Training**:
```python
# Chronological split (walk-forward)
train_dates: 2023-2024
val_dates: 2024-2025

model.train(X, y, group_sizes, dates, feature_names)
# Outputs: trained LambdaRank model
```

**Inference**:
```python
scores = model.predict(X)  # Higher score = better rank
ranks = model.rank_stocks(df, feature_cols)
# Returns: DataFrame with rank column
```

**Evaluation Metrics**:
- **NDCG@5**: How well does model rank top 5?
- **Rank IC**: Spearman correlation of scores vs actual ranks
- **Hit Rate**: Fraction of positive-return days that model ranks positively

### 5. Model Registry (model_registry.py)

**Purpose**: Runtime switching between legacy LSTM and new ranker

```python
# Train and register
model = RankerModel()
model.train(...)
register_model("market_v2", model, mode="market", version="2.0")

# Load by name
model = load_model("market_v2")

# Get default
default_model = get_default_model(mode="market")

# List all
all_models = list_models(mode="market")
```

**Key Features**:
- Persistent metadata (JSON registry)
- Model versioning
- Both in-memory cache + disk persistence
- Support for multiple model types (LSTM, Ranker)

### 6. Ranking Service (ranking_service.py)

**Purpose**: End-to-end orchestration of the ranking engine

**Pipeline**:
```
1. Load multi-stock data
2. Load macro data
3. Engineer features (stock + macro + interactions)
4. Score all stocks with trained model
5. Rank and generate portfolio
6. Return macro context + recommendations
```

**API**:
```python
service = RankingService(universe="sp500_tech")

# Full market analysis
result = service.analyze_market(top_n=5, bottom_n=5)
# Returns: {long: [...], short: [...], macro_context: {...}}

# Single stock analysis
details = service.get_stock_details("NVDA")
# Returns: rank, features, macro context

# Portfolio
portfolio = service.get_portfolio(df, top_n=5, bottom_n=5)
```

### 7. Backtest Service (backtest_service.py)

**Purpose**: Backtest ranking-based portfolio strategies

**Strategy Types**:
- **Long-only**: Top N stocks
- **Long-short**: Top N long, Bottom N short (market neutral)

**Features**:
- Daily rebalancing
- Transaction costs (0.1%) + slippage (0.05%)
- Regime filtering (reduce leverage in risk-off)
- Metrics: Sharpe, max drawdown, win rate, hit rate

**Example**:
```python
engine = BacktestEngine()
result = engine.backtest_ranking_strategy(
    returns_df=df[["date", "ticker", "return"]],
    ranks_df=ranked[["date", "ticker", "rank"]],
    long_n=5,
    short_n=5,
)
# Returns: {returns: Series, metrics: {sharpe, drawdown, ...}}
```

### 8. API Routes

#### Market Mode Routes (routes/analyze.py)

```
GET  /market/analyze
  → Full market analysis
  → Query: universe, top_n, bottom_n
  → Response: {long, short, macro_context, metrics}

GET  /market/analyze/{ticker}
  → Single stock within market context
  → Response: {ticker, rank, features}

POST /market/train
  → Train new ranker model
  → Body: {universe, model_name, num_boost_rounds}
  → Response: {status, metrics}

POST /market/backtest
  → Backtest portfolio strategy
  → Body: {universe, long_n, short_n, strategy_type}
  → Response: {metrics: {sharpe, max_dd, ...}}
```

#### Legacy Routes (routes/legacy_routes.py)

**Keeps all existing endpoints functional**:
```
GET  /legacy/predict/{ticker}
GET  /legacy/analyze/{ticker}
POST /legacy/train/{ticker}
GET  /legacy/models
GET  /legacy/chart/{ticker}
```

#### Integration (api/main.py)

```python
# Register both route sets
app.include_router(market_router)    # /market/*
app.include_router(legacy_router)    # /legacy/*

# API root shows architecture
GET / → Description of both systems
```

## Training Workflow

### Training a New Market Model

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training script
python run_market_train.py \
  --universe sp500_tech \
  --model-name market_v2.0 \
  --num-rounds 100

# Output:
# - Loads market_dataset.py → 1000s rows, 8+ stocks
# - Loads macro_service.py → gold, yields, VIX
# - Computes features → 40+ features per stock-date
# - Trains LambdaRank → 100 boosting rounds
# - Evaluates → Rank IC: 0.045
# - Registers model → Available in model registry

# 3. Test with API
curl http://localhost:8000/market/analyze

# 4. Backtest
curl -X POST http://localhost:8000/market/backtest \
  -H "Content-Type: application/json" \
  -d '{"long_n": 5, "short_n": 5}'
```

## Backward Compatibility

**Key Principle**: Old code stays intact, new code runs parallel

### What's Kept
- `data/pipeline.py`: Single-stock data loading (used by legacy LSTM)
- `models/lstm.py`: LSTM architecture unchanged
- `models/train.py`: LSTM training logic unchanged
- All checkpoints: LSTM models still load with `_lstm.pt` suffix

### What's New
- `services/market_dataset.py`: Multi-stock (doesn't touch old code)
- `services/macro_service.py`: Macro indicators (new dependency)
- `services/feature_engineering.py`: New features (new package)
- `models/ranker.py`: LambdaRank (new model type)
- `routes/analyze.py`: Market endpoints (new routes)
- `routes/legacy_routes.py`: Legacy endpoints (new routes)

### Accessing Both Systems

```
Frontend UI:
  /market      → Uses GET /market/analyze
  /legacy      → Uses GET /legacy/analyze/{ticker}
  /            → Uses GET / (shows both options)

API Direct:
  Market: curl http://localhost:8000/market/analyze
  Legacy: curl http://localhost:8000/legacy/predict/AAPL
```

## Configuration

### Market Configuration (config/market_config.py)

Central place to adjust defaults:

```python
DEFAULT_UNIVERSE = "sp500_tech"
INCLUDE_TECHNICAL = True
USE_GOLD = True
USE_YIELDS = True
USE_VIX = True

DEFAULT_LONG_N = 5
DEFAULT_SHORT_N = 5

TRANSACTION_COST = 0.001  # 0.1%
SLIPPAGE = 0.0005        # 0.05%
```

## Performance Expectations

### Rank IC (Evaluation Metric)

What is it? Spearman correlation between predicted ranks and actual returns.

Typical ranges:
- IC < 0.01: Useless (random)
- IC 0.01-0.03: Marginal signal (exists but noisy)
- IC 0.03-0.05: Meaningful signal (competitive)
- IC 0.05-0.10: Strong signal (rare in practice)
- IC > 0.10: Exceptional (verify for overfitting)

Current implementation targets IC ~0.04-0.06 range.

### Sharpe Ratio (Backtest)

Ratio of excess return to volatility.

Typical ranges:
- Sharpe < 0.5: Not viable
- Sharpe 0.5-1.0: Marginal (noisy)
- Sharpe 1.0-2.0: Good (competitive)
- Sharpe > 2.0: Excellent (verify for survivorship bias)

Current implementation targets Sharpe ~1.0-1.5 after costs.

## Next Steps

1. **Run training**: `python run_market_train.py`
2. **Test market analysis**: `curl http://localhost:8000/market/analyze`
3. **Backtest strategy**: `curl -X POST http://localhost:8000/market/backtest`
4. **Analyze single stock**: `curl http://localhost:8000/market/analyze/NVDA`
5. **Compare with legacy**: `curl http://localhost:8000/legacy/predict/NVDA`

## Troubleshooting

### LightGBM Import Error
```
pip install lightgbm>=4.0.0
```

### No market model trained yet
```
python run_market_train.py --universe sp500_tech
```

### API not reflecting changes
```
# Restart API server
uvicorn api.main:app --reload
```

### Memory issues with large universes
- Reduce universe size
- Reduce historical lookback window
- Use monthly data instead of daily

## References

- LambdaRank Paper: "Learning to Rank with Non-Smooth Cost Functions" (Burges et al., 2006)
- Cross-Sectional Momentum: "Profiting from Momentum and Mean Reversion in Stocks" (Lehmann & Modest, 1987)
- Macro Regime Switching: "Regime Shifts: Implications for Dynamic Strategies" (Hamilton, 1989)
- Information Coefficient: "The Grinold-Kahn Fundamental Law of Active Management"
"""
