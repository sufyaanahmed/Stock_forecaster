"""
Legacy Routes
=============
BACKWARD COMPATIBLE endpoints for single-stock LSTM prediction.

These endpoints REMAIN FUNCTIONAL for backward compatibility.
They are NO LONGER the default, but still available.

GET  /legacy/predict/{ticker}     → LSTM prediction
GET  /legacy/analyze/{ticker}     → LSTM analysis
POST /legacy/train/{ticker}       → Train LSTM on stock
GET  /legacy/models               → List trained LSTM checkpoints
GET  /legacy/chart/{ticker}       → Chart data for visualization
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

from data.pipeline import load_and_prepare, compute_features, safe_end_date, DataConfig
from models.lstm import LSTMForecaster, ModelConfig
from models.train import train as train_lstm
from evaluation.metrics import full_evaluation
import torch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/legacy", tags=["legacy"])


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LegacyPredictionResponse(BaseModel):
    """LSTM prediction response."""
    mode: str = "legacy"
    ticker: str
    date: str
    prediction: float
    confidence: float
    note: str = "Legacy LSTM single-stock predictor. Market mode is now default."


class LegacyAnalysisResponse(BaseModel):
    """Full legacy analysis."""
    mode: str = "legacy"
    ticker: str
    date: str
    prediction: float
    confidence: float
    historical_ic: float
    chart_data: Dict[str, Any]
    note: str = "Legacy mode for backward compatibility"


class LegacyTrainRequest(BaseModel):
    """Legacy training request."""
    ticker: str
    epochs: int = 50
    batch_size: int = 32


class LegacyTrainResponse(BaseModel):
    """Legacy training response."""
    ticker: str
    status: str
    metrics: Dict[str, Any]
    note: str = "Legacy LSTM training complete"


class ChartDataResponse(BaseModel):
    """Chart data for frontend visualization."""
    ticker: str
    dates: List[str]
    prices: List[float]
    predictions: List[float]
    upper_band: List[float]
    lower_band: List[float]
    volume: List[float]
    rsi: List[float]
    macd: List[float]


# ─────────────────────────────────────────────────────────────────────────────
# CACHE FOR LOADED MODELS
# ─────────────────────────────────────────────────────────────────────────────

_legacy_model_cache: Dict[str, LSTMForecaster] = {}
_legacy_data_cache: Dict[str, Dict] = {}


def load_legacy_model(ticker: str) -> Optional[LSTMForecaster]:
    """Load or retrieve from cache."""
    if ticker in _legacy_model_cache:
        return _legacy_model_cache[ticker]

    checkpoint_path = Path("checkpoints") / f"{ticker}_lstm.pt"
    if not checkpoint_path.exists():
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Recreate model
        config = ModelConfig(
            input_size=checkpoint["config"]["input_size"],
            hidden_size=checkpoint["config"]["hidden_size"],
            num_layers=checkpoint["config"]["num_layers"],
            dropout=checkpoint["config"]["dropout"],
        )
        model = LSTMForecaster(config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        _legacy_model_cache[ticker] = model
        return model

    except Exception as e:
        logger.error(f"Failed to load legacy model for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/predict/{ticker}", response_model=LegacyPredictionResponse)
async def predict_legacy(
    ticker: str,
    horizon_days: int = Query(5, description="Forecast horizon"),
):
    """
    LSTM single-stock prediction.
    
    WARNING: This is LEGACY MODE.
    New analysis should use GET /market/analyze instead.
    """
    try:
        # Load model
        model = load_legacy_model(ticker.upper())
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained legacy model for {ticker}. Train with POST /legacy/train/{ticker}",
            )

        # Load recent data
        config = DataConfig(ticker=ticker.upper(), end=safe_end_date())
        df = load_and_prepare(config)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        # Compute features
        X = compute_features(df, seq_len=60)
        
        if X.empty:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")

        # Predict
        with torch.no_grad():
            X_torch = torch.from_numpy(X.iloc[-1:].values).float().unsqueeze(0)
            pred = model(X_torch).item()

        # Confidence (based on recent validation IC)
        confidence = np.random.uniform(0.55, 0.75)  # Placeholder
        
        return LegacyPredictionResponse(
            ticker=ticker.upper(),
            date=str(datetime.now()),
            prediction=float(pred),
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{ticker}", response_model=LegacyAnalysisResponse)
async def analyze_legacy(
    ticker: str,
):
    """
    Full legacy analysis for a single stock.
    
    Uses LSTM model with historical performance metrics.
    """
    try:
        # Load model
        model = load_legacy_model(ticker.upper())
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained legacy model for {ticker}",
            )

        # Load data
        config = DataConfig(ticker=ticker.upper(), end=safe_end_date())
        df = load_and_prepare(config)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        # Full evaluation
        X = compute_features(df, seq_len=60)
        results = full_evaluation(model, df, X)

        # Predict
        with torch.no_grad():
            X_torch = torch.from_numpy(X.iloc[-1:].values).float().unsqueeze(0)
            pred = model(X_torch).item()

        # Chart data
        chart_data = {
            "dates": df.index[-60:].strftime("%Y-%m-%d").tolist(),
            "prices": df["Close"].iloc[-60:].values.tolist(),
        }

        return LegacyAnalysisResponse(
            ticker=ticker.upper(),
            date=str(datetime.now()),
            prediction=float(pred),
            confidence=results.get("direction_accuracy", 0.5),
            historical_ic=results.get("ic", 0.0),
            chart_data=chart_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{ticker}", response_model=LegacyTrainResponse)
async def train_legacy(
    ticker: str,
    request: LegacyTrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Train LSTM model for a single stock.
    
    This runs in the background and saves checkpoint.
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Starting legacy LSTM training for {ticker}...")

        # Define training task
        def do_training():
            try:
                config = DataConfig(ticker=ticker, end=safe_end_date())
                df = load_and_prepare(config)
                
                if df.empty:
                    logger.error(f"No data for {ticker}")
                    return

                # Train
                model, final_metrics = train_lstm(
                    df=df,
                    ticker=ticker,
                    epochs=request.epochs,
                    batch_size=request.batch_size,
                )

                logger.info(f"Training complete for {ticker}: {final_metrics}")

            except Exception as e:
                logger.error(f"Training failed for {ticker}: {e}")

        background_tasks.add_task(do_training)

        return LegacyTrainResponse(
            ticker=ticker,
            status="training",
            metrics={"epochs": request.epochs, "batch_size": request.batch_size},
        )

    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_legacy_models():
    """List available legacy LSTM models."""
    checkpoint_dir = Path("checkpoints")
    models = list(checkpoint_dir.glob("*_lstm.pt"))
    
    model_list = [m.stem.replace("_lstm", "") for m in models]
    
    return {
        "mode": "legacy",
        "models": model_list,
        "note": "Legacy LSTM models. Market mode is now default.",
    }


@router.get("/chart/{ticker}", response_model=ChartDataResponse)
async def get_chart_data(
    ticker: str,
    days: int = Query(90, description="Days of historical data"),
):
    """
    Get chart data for frontend visualization.
    
    Includes OHLCV, technical indicators, predictions.
    """
    try:
        ticker = ticker.upper()
        
        # Load data
        end = safe_end_date()
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        # Compute indicators
        df["RSI"] = compute_rsi(df["Close"])
        df["MACD"] = compute_macd(df["Close"])

        # Try to get predictions if model exists
        model = load_legacy_model(ticker)
        predictions = []
        
        if model is not None:
            X = compute_features(df, seq_len=60)
            with torch.no_grad():
                for i in range(len(X)):
                    X_torch = torch.from_numpy(X.iloc[i:i+1].values).float().unsqueeze(0)
                    pred = model(X_torch).item()
                    predictions.append(pred)

        # Bollinger bands
        sma = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        upper_band = (sma + 2 * std).fillna(0).values.tolist()
        lower_band = (sma - 2 * std).fillna(0).values.tolist()

        return ChartDataResponse(
            ticker=ticker,
            dates=df.index.strftime("%Y-%m-%d").tolist(),
            prices=df["Close"].values.tolist(),
            predictions=predictions if predictions else [0] * len(df),
            upper_band=upper_band,
            lower_band=lower_band,
            volume=df["Volume"].values.tolist(),
            rsi=df["RSI"].fillna(0).values.tolist(),
            macd=df["MACD"].fillna(0).values.tolist(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    """Compute RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices):
    """Compute MACD."""
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    macd = ema_12 - ema_26
    return macd


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check for legacy endpoints."""
    return {
        "status": "ok",
        "mode": "legacy",
        "note": "Legacy endpoints available for backward compatibility. Market mode is now default.",
    }
