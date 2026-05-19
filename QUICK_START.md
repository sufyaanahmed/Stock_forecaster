# QUICK REFERENCE GUIDE

## 🚀 Market Mode (NEW DEFAULT)

### Train a Model
```bash
python run_market_train.py --universe sp500_tech --num-rounds 100
```

### Analyze Market
```bash
curl http://localhost:8000/market/analyze
```

Response:
```json
{
  "mode": "market",
  "date": "2026-05-18T...",
  "macro_context": {
    "gold_trend": "rising",
    "yield_trend": "falling",
    "vix_level": "calm",
    "risk_regime": "risk_on"
  },
  "long": ["NVDA", "META", "AAPL", ...],
  "short": ["TSLA", "AMZN", ...],
  "metrics": {
    "rank_ic": 0.045,
    "top_features": {...}
  }
}
```

### Analyze Single Stock
```bash
curl http://localhost:8000/market/analyze/NVDA
```

### Backtest Strategy
```bash
curl -X POST http://localhost:8000/market/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "long_short",
    "long_n": 5,
    "short_n": 5
  }'
```

Response:
```json
{
  "metrics": {
    "annual_return": 0.18,
    "sharpe": 1.25,
    "max_drawdown": -0.15,
    "win_rate": 0.53
  }
}
```

---

## 🔄 Legacy Mode (BACKWARD COMPATIBLE)

### Predict Stock (Old Way)
```bash
curl http://localhost:8000/legacy/predict/AAPL
```

### Train LSTM
```bash
curl -X POST http://localhost:8000/legacy/train/AAPL \
  -H "Content-Type: application/json" \
  -d '{"epochs": 100}'
```

### Get Chart Data
```bash
curl http://localhost:8000/legacy/chart/AAPL?period=1y
```

---

## 📊 Configuration

Edit `config/market_config.py`:

```python
DEFAULT_UNIVERSE = "sp500_tech"      # Change universe
DEFAULT_LONG_N = 5                   # Change portfolio size
TRANSACTION_COST = 0.001             # Change cost assumption
USE_GOLD = True                      # Include/exclude macro
USE_YIELDS = True
USE_VIX = True
```

---

## 🔧 Common Tasks

### Train Multiple Models
```bash
python run_market_train.py --universe sp500_tech --model-name market_v1
python run_market_train.py --universe nasdaq_100 --model-name nasdaq_v1
```

### List All Models
```bash
from models.model_registry import list_models

models = list_models(mode="market")
for name, meta in models.items():
    print(f"{name}: {meta['metrics']}")
```

### Compare Strategies
```bash
curl -X POST http://localhost:8000/market/backtest \
  -d '{"strategy_type": "long_only"}'     # Long-only

curl -X POST http://localhost:8000/market/backtest \
  -d '{"strategy_type": "long_short"}'    # Market neutral
```

### Debug Individual Stock
```python
from services.ranking_service import RankingService

service = RankingService()
details = service.get_stock_details("NVDA")
print(f"Rank: {details['rank']}")
print(f"Features: {details['features']}")
```

---

## 📈 Interpreting Results

### Rank IC
- **0.01-0.03**: Weak (noisy)
- **0.03-0.05**: Good (competitive)
- **0.05+**: Excellent (verify not overfit)

### Sharpe Ratio
- **< 0.5**: Not viable
- **0.5-1.0**: Marginal
- **1.0-2.0**: Good
- **> 2.0**: Excellent

### Win Rate
- **50-52%**: Random
- **52-54%**: Weak edge
- **54%+**: Meaningful

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| LightGBM not found | `pip install lightgbm>=4.0.0` |
| No market model | `python run_market_train.py` |
| API slow | Reduce universe size |
| Memory error | Use monthly data instead |
| Old endpoints broken | Use `/legacy/*` prefix |

---

## 📚 Documentation

- **Full Guide**: `Readme.md`
- **Architecture**: `ARCHITECTURE.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Code Comments**: See docstrings in each file

---

## 🎯 Feature Flags

Enable/disable features in `config/market_config.py`:

```python
INCLUDE_TECHNICAL = True       # RSI, MACD, Bollinger
INCLUDE_VOLUME = True          # Volume ratios
INCLUDE_GAPS = True            # Overnight gaps
INCLUDE_INTERACTIONS = True    # Stock × macro
USE_GOLD = True
USE_YIELDS = True
USE_VIX = True
```

---

## 💡 Pro Tips

1. **Start small**: Train on `sp500_tech` universe first
2. **Monitor IC**: Rank IC 0.03-0.05 is healthy
3. **Check regimes**: Market works best in trending markets
4. **Compare**: Always backtest both long_only and long_short
5. **Update**: Retrain weekly/monthly with new data
6. **Diversify**: Use multiple universes, not just one
7. **Document**: Log which model performed best

---

## 🚀 Next Steps

1. ✅ Install: `pip install -r requirements.txt`
2. ✅ Train: `python run_market_train.py`
3. ✅ Test: `curl http://localhost:8000/market/analyze`
4. ✅ Backtest: `curl -X POST http://localhost:8000/market/backtest`
5. ✅ Deploy: Point frontend to `/market/*` endpoints

---

**Questions?** See `ARCHITECTURE.md` for detailed explanations.
