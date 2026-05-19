"""
FastAPI Backend — QuantML Stock Forecaster
==========================================
Endpoints:
  GET  /api/health              → liveness probe
  GET  /api/models              → list trained checkpoints
  GET  /api/chart/{ticker}      → OHLCV + indicators (chart data)
  GET  /api/analyze/{ticker}    → full analysis + prediction
  POST /api/train               → trigger background training
  GET  /api/train/status/{tick} → poll training progress

Design:
  - Server-side data cache (TTL-based) avoids hammering yfinance on every request
  - Model cache keeps loaded PyTorch models in memory (no disk re-load per request)
  - Background tasks run training without blocking the event loop
  - CORS is permissive in dev; the Vite proxy means the browser never sees it
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path so imports find local packages regardless of cwd ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import (
    DataConfig,
    compute_features,
    load_and_prepare,
    safe_end_date,
)
from evaluation.metrics import backtest, full_evaluation
from models.lstm import LSTMForecaster, ModelConfig
from models.train import train
from services.log_service import install_handler, get_logs, push_log

# ── Import new routes (Market Mode + Legacy Mode) ────────────────────────────
from routes.analyze import router as market_router
from routes.legacy_routes import router as legacy_router

# ── Install centralized log handler on all relevant loggers ──────────────────
for _logger_name in ["", "services", "routes", "data", "models", "evaluation"]:
    install_handler(_logger_name)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="QuantML Stock Forecaster API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Dev: Vite proxy handles CORS; production should restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register NEW market mode routes ──────────────────────────────────────────
app.include_router(market_router)

# ── Register LEGACY routes (backward compatible) ──────────────────────────────
app.include_router(legacy_router)

CHECKPOINT_DIR = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global mode state ─────────────────────────────────────────────────────────
# Only ONE pipeline is active at a time.
_active_mode: str = "legacy"   # "legacy" | "market"

# ── Server-side caches ────────────────────────────────────────────────────────

# Model cache: ticker → {model, feature_cols, test_metrics, history}
_model_cache: Dict[str, Any] = {}

# Data cache: ticker → {data_dict, fetched_at}
# Avoids re-downloading/re-preprocessing on back-to-back requests
_data_cache: Dict[str, Dict] = {}
DATA_CACHE_TTL = 300  # seconds (5 min)

# Chart-specific raw data cache: key → {payload, fetched_at}
_chart_cache: Dict[str, Dict] = {}
CHART_CACHE_TTL = 120  # seconds (2 min)

# ── Schemas ───────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker: str
    start: str = "2015-01-01"
    epochs: int = 100
    seq_len: int = 60


class AnalysisResponse(BaseModel):
    ticker: str
    latest_close: float
    predicted_direction: str        # "UP" | "DOWN" | "NEUTRAL"
    predicted_return_pct: float
    confidence_score: float
    ic: float
    ic_significant: bool
    direction_accuracy: float
    sharpe: float
    max_drawdown: float
    total_return: float
    buy_hold_return: float
    model_trained: bool
    data_source: str = "yfinance"
    cache_hit: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_prepared_data(ticker: str, start: str = "2015-01-01", seq_len: int = 60) -> dict:
    """
    Load and prepare data with server-side TTL cache.
    Avoids re-downloading yfinance data on every request.
    """
    cache_key = f"{ticker}:{start}:{seq_len}"
    entry = _data_cache.get(cache_key)
    if entry and (time.time() - entry["fetched_at"]) < DATA_CACHE_TTL:
        return entry["data"]

    cfg  = DataConfig(ticker=ticker, start=start, seq_len=seq_len)
    data = load_and_prepare(cfg)
    _data_cache[cache_key] = {"data": data, "fetched_at": time.time()}
    return data


def _load_model(ticker: str) -> Optional[dict]:
    """Load PyTorch model from checkpoint with in-memory caching."""
    if ticker in _model_cache:
        return _model_cache[ticker]

    path = Path(CHECKPOINT_DIR) / f"{ticker}_lstm.pt"
    if not path.exists():
        return None

    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        cfg   = checkpoint["model_config"]
        model = LSTMForecaster(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        _model_cache[ticker] = {
            "model":        model,
            "feature_cols": checkpoint.get("feature_cols", []),
            "test_metrics": checkpoint["test_metrics"],
            "history":      checkpoint.get("history", {}),
        }
        return _model_cache[ticker]
    except Exception as exc:
        print(f"[load_model] ERROR loading {ticker}: {exc}")
        return None


def _safe_series_list(series: pd.Series) -> list:
    """Convert pandas Series to JSON-safe list (NaN → None)."""
    return [
        None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6)
        for v in series
    ]


# ── Training state ────────────────────────────────────────────────────────────
training_status: Dict[str, str] = {}


def _run_training(req: TrainRequest) -> None:
    """Background training job. Updates training_status in-place."""
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
        model  = LSTMForecaster(model_cfg)
        result = train(
            model, data,
            epochs=req.epochs,
            save_dir=CHECKPOINT_DIR,
            ticker=ticker,
        )

        # Invalidate caches so next request picks up fresh model + data
        _model_cache.pop(ticker, None)
        _data_cache.pop(f"{ticker}:{req.start}:{req.seq_len}", None)

        ic = result["test_metrics"]["ic"]
        training_status[ticker] = f"done|IC:{ic:.4f}"
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[training] ERROR for {ticker}:\n{tb}")
        training_status[ticker] = f"error|{exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """
    API root endpoint.
    
    SYSTEM UPGRADE: This platform now supports BOTH:
    
    1. NEW DEFAULT - Market Mode (Macro-aware Multi-Stock Ranking):
       - GET  /market/analyze              → Full market analysis
       - GET  /market/analyze/{ticker}     → Single stock within market context
       - POST /market/train                → Train LambdaRank model
       - POST /market/backtest             → Backtest strategy
    
    2. LEGACY - Single-Stock LSTM (Backward Compatible):
       - GET  /legacy/predict/{ticker}     → LSTM prediction
       - GET  /legacy/analyze/{ticker}     → LSTM analysis
       - POST /legacy/train/{ticker}       → Train LSTM
       - GET  /legacy/models               → List LSTM models
       - GET  /legacy/chart/{ticker}       → Chart data
    
    Note: Old single-stock endpoints (GET /api/analyze/{ticker}, etc.) remain functional.
    """
    return {
        "title": "QuantML Stock Forecaster - UPGRADED ARCHITECTURE",
        "version": "2.0",
        "mode": "dual (legacy + market)",
        "default_system": "market",
        "market_endpoints": "/market/*",
        "legacy_endpoints": "/legacy/*",
        "status": "The platform now supports both single-stock LSTM (legacy) and macro-aware multi-stock ranking (market). Market mode is the new default.",
    }


@app.get("/api/health")
def health():
    """Liveness probe. Returns immediately."""
    return {"status": "ok", "device": str(device), "ts": int(time.time()), "mode": _active_mode}


# ── Mode switching ────────────────────────────────────────────────────────────

class ModeRequest(BaseModel):
    mode: str   # "legacy" | "market"


@app.get("/api/mode")
def get_mode():
    """Return the currently active pipeline mode."""
    return {"mode": _active_mode}


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    """Switch between 'legacy' (single-stock LSTM) and 'market' (ranker) modes."""
    global _active_mode
    if req.mode not in ("legacy", "market"):
        raise HTTPException(status_code=400, detail="mode must be 'legacy' or 'market'")
    _active_mode = req.mode
    push_log("INFO", "api", f"Mode switched to '{_active_mode}'")
    return {"mode": _active_mode, "status": "ok"}


# ── Logs endpoint (polling-based SSE alternative) ─────────────────────────────

@app.get("/api/logs")
def get_log_records(
    since: int = Query(0, description="Return only records after this epoch-ms timestamp"),
    limit: int = Query(200, ge=1, le=500),
):
    """Return recent structured log records for frontend live-log panel."""
    return {"logs": get_logs(since_ts=since, limit=limit)}


@app.get("/api/models")
def list_models():
    """Return all trained model checkpoints with their test metrics."""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    result = []
    for f in sorted(Path(CHECKPOINT_DIR).glob("*_lstm.pt")):
        ticker = f.stem.replace("_lstm", "").upper()
        try:
            ckpt = torch.load(str(f), map_location="cpu", weights_only=False)
            result.append({
                "ticker":       ticker,
                "test_ic":      round(ckpt["test_metrics"]["ic"],      4),
                "test_dir_acc": round(ckpt["test_metrics"]["dir_acc"], 4),
            })
        except Exception:
            pass   # skip corrupt checkpoints silently
    return result


@app.post("/api/train")
def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start training in the background.
    Poll /api/train/status/{ticker} for progress.
    """
    ticker = req.ticker.upper()
    if training_status.get(ticker) == "running":
        return {"status": "already_running", "ticker": ticker}
    background_tasks.add_task(_run_training, req)
    return {"status": "started", "ticker": ticker}


@app.get("/api/train/status/{ticker}")
def train_status(ticker: str):
    """Return current training status for a ticker."""
    status = training_status.get(ticker.upper(), "not_started")
    return {"ticker": ticker.upper(), "status": status}


@app.get("/api/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "1y"):
    """
    OHLCV + technical indicators for the frontend charting panel.
    Periods: 3m | 6m | 1y | 2y | 5y

    Uses a server-side 2-minute cache to avoid hammering yfinance.
    Fetches extra warm-up rows (90 days) so rolling indicators are valid.
    """
    cache_key = f"chart:{ticker}:{period}"
    entry = _chart_cache.get(cache_key)
    if entry and (time.time() - entry["fetched_at"]) < CHART_CACHE_TTL:
        payload = dict(entry["payload"])
        payload["cache_hit"] = True
        return payload

    period_days = {"3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825}
    days      = period_days.get(period, 365)
    warmup    = 90  # extra days for rolling windows to warm up
    total_days = days + warmup

    import datetime
    today  = datetime.datetime.utcnow()
    start  = (today - datetime.timedelta(days=total_days)).strftime("%Y-%m-%d")
    end    = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker.upper(),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty or len(raw) < 30:
            raise ValueError(
                f"Not enough data returned for {ticker.upper()} "
                f"({0 if raw is None else len(raw)} rows)"
            )

        # Flatten MultiIndex columns (yfinance ≥0.2.x)
        raw.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in raw.columns
        ]
        raw = raw.dropna()
        raw.index = pd.to_datetime(raw.index)

        feat = compute_features(raw)

        # Trim to requested period (discard warm-up rows)
        cutoff = today - datetime.timedelta(days=days)
        feat   = feat[feat.index >= pd.Timestamp(cutoff)]
        raw    = raw.loc[feat.index]

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = {
        "ticker":   ticker.upper(),
        "period":   period,
        "dates":    [str(d.date()) for d in feat.index],
        "ohlcv": {
            "open":   _safe_series_list(raw["open"]),
            "high":   _safe_series_list(raw["high"]),
            "low":    _safe_series_list(raw["low"]),
            "close":  _safe_series_list(raw["close"]),
            "volume": _safe_series_list(raw["volume"]),
        },
        "indicators": {
            "rsi_14":      _safe_series_list(feat["rsi_14"]),
            "macd_norm":   _safe_series_list(feat["macd_norm"]),
            "bb_position": _safe_series_list(feat["bb_position"]),
            "vol_20":      _safe_series_list(feat["vol_20"]),
        },
        "log_returns": _safe_series_list(feat["log_return"]),
        "cache_hit":   False,
    }

    _chart_cache[cache_key] = {"payload": payload, "fetched_at": time.time()}
    return payload


@app.get("/api/analyze/{ticker}", response_model=AnalysisResponse)
def analyze_ticker(ticker: str):
    """
    Full analysis pipeline:
      1. Load or fetch prepared data (cached 5 min)
      2. Load model from checkpoint (cached in memory)
      3. Run inference on test set
      4. Compute backtest metrics
      5. Return structured response

    If no model checkpoint exists, returns model_trained=False with price only.
    """
    ticker = ticker.upper()
    cache_hit = False

    # ── 1. Data ──────────────────────────────────────────────────────────────
    try:
        cache_key = f"{ticker}:2015-01-01:60"
        entry = _data_cache.get(cache_key)
        if entry and (time.time() - entry["fetched_at"]) < DATA_CACHE_TTL:
            data      = entry["data"]
            cache_hit = True
        else:
            data = _get_prepared_data(ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data pipeline error: {exc}")

    # ── 2. Latest close price ─────────────────────────────────────────────────
    try:
        raw = data.get("raw")
        if raw is not None and not raw.empty:
            latest_close = float(raw["close"].iloc[-1])
        else:
            latest_close = 0.0
    except Exception:
        latest_close = 0.0

    # ── 3. Load model ─────────────────────────────────────────────────────────
    loaded = _load_model(ticker)

    if loaded is None:
        # No checkpoint → return price + untrained placeholder
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
            cache_hit=cache_hit,
        )

    # ── 4. Inference ─────────────────────────────────────────────────────────
    model  = loaded["model"]
    X_test = torch.FloatTensor(data["X_test"]).to(device)
    y_test = data["y_test"]

    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()

    latest_pred = float(y_pred[-1])
    if   latest_pred >  0.0005: direction = "UP"
    elif latest_pred < -0.0005: direction = "DOWN"
    else:                       direction = "NEUTRAL"

    abs_preds  = np.abs(y_pred)
    confidence = float(np.mean(abs_preds <= abs(latest_pred)))

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    bt          = backtest(y_test, y_pred)
    eval_result = full_evaluation(y_test, y_pred, ticker=ticker, verbose=False)

    return AnalysisResponse(
        ticker=ticker,
        latest_close=latest_close,
        predicted_direction=direction,
        predicted_return_pct=round(latest_pred * 100, 4),
        confidence_score=round(confidence, 3),
        ic=eval_result["ic"],
        ic_significant=eval_result["ic_significant"],
        direction_accuracy=eval_result["direction_acc"],
        sharpe=bt["sharpe"],
        max_drawdown=bt["max_drawdown"],
        total_return=bt["total_return"],
        buy_hold_return=bt["buy_hold_return"],
        model_trained=True,
        cache_hit=cache_hit,
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "data", "models", "evaluation"],
    )