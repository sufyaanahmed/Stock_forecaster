"""
FastAPI Backend
===============
Serves:
  - /api/analyze/{ticker}    : full analysis (features, prediction, backtest stats)
  - /api/predict/{ticker}    : latest signal only (fast)
  - /api/chart/{ticker}      : OHLCV + indicators for frontend chart
  - /api/train/{ticker}      : trigger model training (async)
  - /api/models              : list trained models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import os, json
from pathlib import Path
from typing import Optional, Dict, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.pipeline import DataConfig, load_and_prepare, fetch_raw, compute_features
from models.lstm import LSTMForecaster, ModelConfig
from models.train import train, information_coefficient, direction_accuracy
from evaluation.metrics import full_evaluation, backtest, compute_sharpe

app = FastAPI(title="Stock Forecaster API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_DIR = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory cache for loaded models (avoid re-loading on every request)
_model_cache: Dict[str, Any] = {}


# ── Schema ────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker: str
    start: str = "2015-01-01"
    epochs: int = 100
    seq_len: int = 60


class AnalysisResponse(BaseModel):
    ticker: str
    latest_close: float
    predicted_direction: str       # "UP" | "DOWN" | "NEUTRAL"
    predicted_return_pct: float
    confidence_score: float        # |predicted_return| scaled 0-1 relative to history
    ic: float
    ic_significant: bool
    direction_accuracy: float
    sharpe: float
    max_drawdown: float
    total_return: float
    buy_hold_return: float
    model_trained: bool


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ticker: str):
    """Load model from checkpoint, with in-memory caching."""
    if ticker in _model_cache:
        return _model_cache[ticker]

    path = f"{CHECKPOINT_DIR}/{ticker}_lstm.pt"
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["model_config"]
    model = LSTMForecaster(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model_cache[ticker] = {
        "model": model,
        "feature_cols": checkpoint["feature_cols"],
        "test_metrics": checkpoint["test_metrics"],
        "history": checkpoint["history"],
    }
    return _model_cache[ticker]


# ── Background training ───────────────────────────────────────────────────────

training_status: Dict[str, str] = {}


def run_training(req: TrainRequest):
    """Run in background — doesn't block the API."""
    ticker = req.ticker.upper()
    training_status[ticker] = "running"
    try:
        cfg = DataConfig(ticker=ticker, start=req.start, seq_len=req.seq_len)
        data = load_and_prepare(cfg)
        
        model_cfg = ModelConfig(
            input_size=data["X_train"].shape[2],
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            use_attention=True,
        )
        model = LSTMForecaster(model_cfg)
        result = train(
            model, data,
            epochs=req.epochs,
            save_dir=CHECKPOINT_DIR,
            ticker=ticker,
        )
        # Invalidate cache so fresh model loads next request
        _model_cache.pop(ticker, None)
        training_status[ticker] = f"done|IC:{result['test_metrics']['ic']:.4f}"
    except Exception as e:
        training_status[ticker] = f"error|{str(e)}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/api/train")
def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start training in background. Poll /api/train/status/{ticker} for progress."""
    ticker = req.ticker.upper()
    if training_status.get(ticker) == "running":
        return {"status": "already_running", "ticker": ticker}
    background_tasks.add_task(run_training, req)
    return {"status": "started", "ticker": ticker}


@app.get("/api/train/status/{ticker}")
def train_status(ticker: str):
    status = training_status.get(ticker.upper(), "not_started")
    return {"ticker": ticker.upper(), "status": status}


@app.get("/api/models")
def list_models():
    """List all trained models."""
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    models = []
    for f in Path(CHECKPOINT_DIR).glob("*_lstm.pt"):
        ticker = f.stem.replace("_lstm", "")
        ckpt = torch.load(f, map_location="cpu")
        models.append({
            "ticker": ticker,
            "test_ic": round(ckpt["test_metrics"]["ic"], 4),
            "test_dir_acc": round(ckpt["test_metrics"]["dir_acc"], 4),
        })
    return models


@app.get("/api/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "1y"):
    """
    OHLCV + technical indicators for the frontend chart.
    period: '3m' | '6m' | '1y' | '2y' | '5y'
    """
    period_map = {"3m": "90d", "6m": "180d", "1y": "365d", "2y": "730d", "5y": "1825d"}
    days = int(period_map.get(period, "365d").replace("d", ""))
    
    import datetime
    start = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    try:
        cfg = DataConfig(ticker=ticker.upper(), start=start)
        raw = fetch_raw(cfg)
        feat = compute_features(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Align raw and features (feat drops first N rows due to rolling windows)
    raw_aligned = raw.loc[feat.index]

    def safe_list(series):
        return [round(float(v), 4) if not np.isnan(v) else None for v in series]

    return {
        "ticker": ticker.upper(),
        "dates":  [str(d.date()) for d in feat.index],
        "ohlcv": {
            "open":   safe_list(raw_aligned['open']),
            "high":   safe_list(raw_aligned['high']),
            "low":    safe_list(raw_aligned['low']),
            "close":  safe_list(raw_aligned['close']),
            "volume": safe_list(raw_aligned['volume']),
        },
        "indicators": {
            "rsi_14":      safe_list(feat['rsi_14']),
            "macd_norm":   safe_list(feat['macd_norm']),
            "bb_position": safe_list(feat['bb_position']),
            "vol_20":      safe_list(feat['vol_20']),
        },
        "log_returns": safe_list(feat['log_return']),
    }


@app.get("/api/analyze/{ticker}", response_model=AnalysisResponse)
def analyze_ticker(ticker: str):
    """
    Full analysis: load model → latest features → predict → backtest stats.
    If no model trained yet, returns stats with model_trained=False.
    """
    ticker = ticker.upper()
    loaded = load_model(ticker)

    try:
        cfg = DataConfig(ticker=ticker)
        data = load_and_prepare(cfg)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data error: {e}")

    raw = fetch_raw(cfg)
    latest_close = float(raw['close'].iloc[-1])

    if loaded is None:
        return AnalysisResponse(
            ticker=ticker,
            latest_close=latest_close,
            predicted_direction="NEUTRAL",
            predicted_return_pct=0.0,
            confidence_score=0.0,
            ic=0.0,
            ic_significant=False,
            direction_accuracy=0.5,
            sharpe=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            buy_hold_return=0.0,
            model_trained=False,
        )

    model = loaded["model"]

    # Get latest sequence for live prediction
    X_test = torch.FloatTensor(data["X_test"]).to(device)
    y_test = data["y_test"]

    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()

    latest_pred = float(y_pred[-1])
    direction = "UP" if latest_pred > 0.0005 else ("DOWN" if latest_pred < -0.0005 else "NEUTRAL")

    # Confidence: where does |latest_pred| sit in distribution of |predictions|
    abs_preds = np.abs(y_pred)
    confidence = float(np.mean(abs_preds <= abs(latest_pred)))

    # Backtest on test set
    bt = backtest(y_test, y_pred)
    eval_result = full_evaluation(y_test, y_pred, ticker=ticker, verbose=False)

    return AnalysisResponse(
        ticker=ticker,
        latest_close=latest_close,
        predicted_direction=direction,
        predicted_return_pct=round(latest_pred * 100, 4),
        confidence_score=round(confidence, 3),
        ic=eval_result["ic"],
        ic_significant=eval_result["ic_significant"],
        direction_accuracy=eval_result["direction_accuracy"],
        sharpe=bt["sharpe"],
        max_drawdown=bt["max_drawdown"],
        total_return=bt["total_return"],
        buy_hold_return=bt["buy_hold_return"],
        model_trained=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)