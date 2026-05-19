# Stock Forecaster — ML Research Project

**MAJOR UPGRADE**: This platform now supports **DUAL MODES**:

## 🚀 NEW: Market Mode (Default) — Macro-Aware Multi-Stock Ranking

A macro-regime-aware cross-sectional ranking engine that compares stocks **relative to each other**
using LightGBM LambdaRank. Replaces single-stock forecasting with portfolio intelligence.

### What's New

- **Multi-stock ranking** (not single-stock prediction)
- **Macro integration** (gold, yields, VIX, fed rates)
- **Cross-sectional features** (relative strength, interactions)
- **LambdaRank model** (learns "which stocks outperform peers?")
- **Portfolio construction** (long/short candidates + risk regime filtering)
- **Backtesting** (daily rebalancing, transaction costs, Sharpe/drawdown)

### Quick Start: Market Mode

```bash
# 1. Install (includes lightgbm)
pip install -r requirements.txt

# 2. Train market model
curl -X POST http://localhost:8000/market/train \
  -H "Content-Type: application/json" \
  -d '{"universe": "sp500_tech", "model_name": "market_v2"}'

# 3. Analyze market
curl http://localhost:8000/market/analyze

# 4. Analyze single stock in market context
curl http://localhost:8000/market/analyze/NVDA

# 5. Backtest strategy
curl -X POST http://localhost:8000/market/backtest \
  -H "Content-Type: application/json" \
  -d '{"strategy_type": "long_short", "long_n": 5, "short_n": 5}'
```

### Market Mode API

```
GET  /market/analyze              → Full market analysis + top longs/shorts
GET  /market/analyze/{ticker}     → Stock ranking within market context
POST /market/train                → Train LambdaRank model
POST /market/backtest             → Backtest portfolio strategy
GET  /market/health               → Health check
```

---

## Legacy: Single-Stock LSTM (Backward Compatible)

The original LSTM system remains **fully functional** for backward compatibility.
It is now available under `/legacy/` routes but **NOT the default**.

### Legacy Mode API

```
GET  /legacy/predict/{ticker}     → LSTM single-stock prediction
GET  /legacy/analyze/{ticker}     → LSTM analysis with metrics
POST /legacy/train/{ticker}       → Train LSTM on stock
GET  /legacy/models               → List trained LSTM checkpoints
GET  /legacy/chart/{ticker}       → Chart data for visualization
GET  /legacy/health               → Health check
```

### Quick Start: Legacy Mode

```bash
# 1. Train LSTM (background)
curl -X POST http://localhost:8000/legacy/train/AAPL \
  -H "Content-Type: application/json" \
  -d '{"epochs": 100}'

# 2. Get prediction
curl http://localhost:8000/legacy/predict/AAPL

# 3. Full analysis
curl http://localhost:8000/legacy/analyze/AAPL

# 4. Chart data
curl http://localhost:8000/legacy/chart/AAPL?period=1y
```

---

## Architecture (Dual System)

```
MARKET MODE (NEW DEFAULT):
  services/market_dataset.py     → Multi-stock data loading (aligned dates)
  services/macro_service.py      → Gold, yields, VIX, fed rates
  services/feature_engineering.py → 40+ stock + macro interaction features
  models/ranker.py               → LightGBM LambdaRank cross-sectional ranking
  models/model_registry.py       → Model switching (legacy vs market)
  services/ranking_service.py    → Orchestrates full pipeline
  services/backtest_service.py   → Portfolio backtesting with regimes
  routes/analyze.py              → Market analysis endpoints

LEGACY MODE (BACKWARD COMPATIBLE):
  data/pipeline.py               → Single-stock ADF-verified features
  models/lstm.py                 → PyTorch LSTM + attention
  models/train.py                → IC-optimized training
  evaluation/metrics.py          → Backtest engine
  routes/legacy_routes.py        → Single-stock endpoints

SHARED:
  api/main.py                    → FastAPI root + integration
  config/market_config.py        → Central configuration
  frontend/                      → React dashboard (works with both modes)
```

---

## Market Mode Deep Dive

### The Ranking Problem (Different from Prediction)

**OLD** (Single-Stock Prediction):
```
Input:  NVDA's historical features
Output: Tomorrow's NVDA return (+0.5% or -0.3%)
Problem: Treats stocks in isolation
```

**NEW** (Cross-Sectional Ranking):
```
Input:  All stocks' features + macro context
Output: Ranking scores for all stocks
        Interpret as: "Which stocks outperform peers tomorrow?"
Problem: Learns market structure, not isolated forecasts
```

### Features Generated

**Stock Features** (Technical):
- Returns (1d, 5d, 20d), volatility (10, 60), momentum
- RSI, MACD, Bollinger position, volume ratios
- Overnight gaps, relative strength vs universe
- ~25 features per stock

**Macro Features** (Economic):
- Gold returns & volatility
- Yield changes & momentum
- VIX level & regime
- Fed rate effects

**Interaction Features** (Stock × Macro):
- return × yield_change (growth sensitivity)
- volatility × VIX (clustering effects)
- momentum × yield_momentum (macro drift exposure)
- ~10 interaction features

### Target: Cross-Sectional Percentile Rank

For each date:
```
date       | ticker | forward_return | target_rank (0-1)
2026-05-18 | NVDA   | +2.5%          | 0.92 (top performer)
2026-05-18 | AAPL   | -0.5%          | 0.31
2026-05-18 | META   | +1.2%          | 0.71
```

Model learns: "Under these macro conditions + stock features, which stocks rank highest?"

### LambdaRank Objective

```
Objective: lambdarank (not regression)
Metric:    NDCG@5 (top-5 ranking accuracy)
Groups:    Trading dates (all stocks on same day form a group)
Constraint: Chronological train/test split (no look-ahead bias)
```

Why LambdaRank? Because we care about **ranking order**, not absolute values.
A model that ranks top 5 correctly but gets magnitudes wrong is better than
one that predicts magnitudes perfectly but ranks wrong.

### Portfolio Construction

```python
# On each date:
ranked = model.rank_all_stocks(date)
long_portfolio = ranked.top(5)        # Best ranked
short_portfolio = ranked.bottom(5)    # Worst ranked

# 50% capital long, 50% capital short (market neutral approach)
# Daily rebalancing
# Transaction costs: 0.1% per trade + 0.05% slippage
```

### Regime Filtering

```
if risk_regime == "risk_off":
    gross_leverage *= 0.5  # Reduce positions when yields spike/VIX high
elif risk_regime == "neutral":
    gross_leverage *= 1.0
else:  # risk_on
    gross_leverage *= 1.0
```

---

## Original LSTM Architecture (Legacy)

A research-grade single-stock forecasting system using PyTorch LSTMs with
attention, proper leak-free feature engineering, and a FastAPI + React frontend.

### Architecture

```
data/pipeline.py      →  ADF-verified features, leak-free splits, RobustScaler
models/lstm.py        →  PyTorch LSTM + additive attention
models/train.py       →  IC-optimized training, early stopping, gradient clipping
evaluation/metrics.py →  Sharpe, Sortino, IC significance, backtest engine
api/main.py           →  FastAPI serving predictions + chart data
frontend/             →  React dashboard with live charts
```

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Train original LSTM (legacy mode)
python run_trains.py --ticker AAPL
python run_trains.py --ticker TSLA --epochs 150
python run_trains.py --ticker RELIANCE.NS --start 2018-01-01

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173

# 5. Access the platform
# Market mode:  http://localhost:5173/market
# Legacy mode:  http://localhost:5173/legacy
# Dashboard:    http://localhost:5173
```

## Key Design Decisions

### Why log returns, not prices?
Prices are non-stationary (unit root). Log returns are stationary, additive
across time, and approximately normal. ADF test verifies this automatically.

### Why IC, not MSE?
In trading, direction matters more than magnitude. IC (Spearman rank correlation)
measures whether the model's ranking of predicted returns matches actual returns.
IC of 0.03–0.05 is considered meaningful in practice.

### Why RobustScaler?
Financial returns have fat tails. RobustScaler uses median/IQR instead of
mean/std, making it resistant to outlier moves (earnings surprises, crashes).

### Why attention on LSTM?
Allows the model to weight which past timesteps matter most for each prediction,
rather than relying on the final hidden state alone. Also improves interpretability.

### Why gradient clipping?
RNNs suffer from exploding gradients. Clipping L2 norm at 1.0 is standard practice.

## Metrics Explained

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| IC (Spearman) | Signal quality vs. actual returns | > 0.03 meaningful |
| Direction Accuracy | % correct up/down calls | > 52% consistently |
| Sharpe Ratio | Risk-adjusted return | > 1.0 acceptable |
| Max Drawdown | Worst peak-to-trough loss | < 20% manageable |
| IC p-value | Statistical significance | < 0.05 to trust IC |

## API Endpoints

```
GET  /api/analyze/{ticker}     Full analysis + model stats
GET  /api/chart/{ticker}       OHLCV + indicators for charts
POST /api/train                Trigger background training
GET  /api/train/status/{ticker} Poll training progress
GET  /api/models               List all trained models
```

## Interview Talking Points

1. **Data leakage prevention**: scaler fit only on train, features computed
   after split, shift(-1) target handled correctly at boundary

2. **Stationarity**: ADF test on target before training, log returns vs prices

3. **Evaluation**: IC + direction accuracy, not just MSE. Backtest with
   realistic transaction costs. Statistical significance testing.

4. **Architecture**: 2-layer LSTM (not 4) to match sample size, attention
   for interpretability, tanh not relu for LSTM activation

5. **Honest limitations**: single-asset, no macro features, no order book data,
   daily frequency limits alpha generation vs HFT strategies