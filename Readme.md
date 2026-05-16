# Stock Forecaster — ML Research Project

A research-grade financial time-series forecasting system using PyTorch LSTMs with
attention, proper leak-free feature engineering, and a FastAPI + React frontend.

## Architecture

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

# 2. Train a model
python run_train.py --ticker AAPL
python run_train.py --ticker TSLA --epochs 150
python run_train.py --ticker RELIANCE.NS --start 2018-01-01   # Indian stocks

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
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