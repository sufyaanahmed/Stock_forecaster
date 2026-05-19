"""
Market Analysis Routes
======================
NEW DEFAULT endpoints for macro-aware multi-stock ranking.

GET  /market/analyze       → Full market analysis
GET  /market/analyze/{ticker}  → Single stock analysis
POST /market/train         → Train ranker model
GET  /market/backtest      → Backtest portfolio strategy
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging

from services.ranking_service import RankingService
from services.backtest_service import BacktestEngine
from data.pipeline import load_and_prepare, compute_features, safe_end_date
from services.market_dataset import MarketDataset, DatasetConfig
from services.macro_service import MacroService, MacroConfig
from services.feature_engineering import FeatureEngineer, FeatureConfig
from models.ranker import RankerModel
from models.model_registry import register_model
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MarketAnalysisResponse(BaseModel):
    """Market analysis response."""
    mode: str = "market"
    date: str
    macro_context: Dict[str, Any]
    long: List[str]
    short: List[str]
    metrics: Dict[str, Any] = {}


class StockAnalysisResponse(BaseModel):
    """Single stock analysis."""
    ticker: str
    date: str
    rank: Optional[float] = None
    features: Dict[str, Any] = {}


class TrainRequest(BaseModel):
    """Training request."""
    universe: str = "sp500_tech"
    model_name: str = "market_v2"
    num_boost_rounds: int = 100


class TrainResponse(BaseModel):
    """Training response."""
    model_name: str
    universe: str
    status: str
    metrics: Dict[str, Any]


class BacktestRequest(BaseModel):
    """Backtest request."""
    universe: str = "sp500_tech"
    long_n: int = 5
    short_n: int = 5
    strategy_type: str = "long_short"  # or "long_only"


class BacktestResponse(BaseModel):
    """Backtest response."""
    universe: str
    strategy_type: str
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# MARKET ANALYSIS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/analyze", response_model=MarketAnalysisResponse)
async def analyze_market(
    universe: str = Query("sp500_tech", description="Stock universe"),
    top_n: int = Query(5, description="Number of long candidates"),
    bottom_n: int = Query(5, description="Number of short candidates"),
):
    """
    Full market analysis with rankings and macro context.
    
    This is the NEW DEFAULT endpoint for market-wide analysis.
    """
    try:
        service = RankingService(universe=universe)
        result = service.analyze_market(
            top_n=top_n,
            bottom_n=bottom_n,
            include_metrics=True,
        )
        return MarketAnalysisResponse(**result)

    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{ticker}", response_model=StockAnalysisResponse)
async def analyze_stock(
    ticker: str,
    universe: str = Query("sp500_tech"),
):
    """
    Analyze a single stock within market context.
    
    Returns ranking and feature contribution within universe.
    """
    try:
        service = RankingService(universe=universe)
        result = service.get_stock_details(ticker)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return StockAnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train_market_model(request: TrainRequest):
    """
    Train a new LambdaRank ranking model.
    
    This trains on the full market dataset with chronological splits.
    Automatically evaluates and registers the model.
    """
    try:
        logger.info(f"Training market model for {request.universe}...")
        
        # Load data
        dataset = MarketDataset(
            DatasetConfig(
                universe=request.universe,
                start_date="2023-01-01",
            )
        )
        stock_df = dataset.build_dataset()
        
        # Macro data
        macro_service = MacroService(
            MacroConfig(
                start_date="2023-01-01",
            )
        )
        macro_df = macro_service.load_all_macro()
        macro_df = macro_service.compute_macro_features(macro_df)
        
        # Features
        engineer = FeatureEngineer(
            FeatureConfig(
                include_technical=True,
                include_volume=True,
                include_gaps=True,
                include_interactions=True,
            )
        )
        feature_df = engineer.engineer_stock_features(stock_df)
        feature_df = engineer.add_macro_interactions(feature_df, macro_df)
        feature_df = engineer.drop_nan_rows(feature_df)
        
        # Prepare for training
        feature_cols = engineer.get_feature_list(include_target=False)
        valid_cols = [c for c in feature_cols if c in feature_df.columns]
        
        X, y, group_sizes, dates = (
            feature_df[valid_cols].fillna(0).values.astype(np.float32),
            feature_df["target_rank"].fillna(0.5).values.astype(np.float32),
            feature_df.groupby("date").size().values,
            feature_df["date"].values,
        )
        
        # Train model
        model = RankerModel()
        model.train(
            X, y, group_sizes, dates,
            feature_names=valid_cols,
            num_boost_round=request.num_boost_rounds,
            train_ratio=0.8,
        )
        
        # Evaluate
        rank_ic = model.eval_rank_ic(feature_df[["date", "ticker"] + valid_cols + ["target_rank"]], valid_cols)
        
        metrics = {
            "rank_ic": float(rank_ic),
            "num_samples": len(feature_df),
            "num_features": len(valid_cols),
            "num_boost_rounds": request.num_boost_rounds,
        }
        
        # Register model
        register_model(
            model_name=request.model_name,
            model_obj=model,
            mode="market",
            version="2.0",
            metrics=metrics,
        )
        
        logger.info(f"Training complete. Rank IC: {rank_ic:.4f}")
        
        return TrainResponse(
            model_name=request.model_name,
            universe=request.universe,
            status="success",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/backtest", response_model=BacktestResponse)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest a ranking-based portfolio strategy.
    
    Evaluates long/short or long-only strategy with:
      - Daily rebalancing
      - Sharpe ratio, drawdown, hit rate
      - Transaction costs and slippage
    """
    try:
        logger.info(f"Backtesting {request.strategy_type} on {request.universe}...")
        
        # Load data and compute returns
        dataset = MarketDataset(
            DatasetConfig(
                universe=request.universe,
                start_date="2022-01-01",  # Longer history for backtest
            )
        )
        stock_df = dataset.build_dataset()
        
        # Compute returns
        stock_df["return"] = stock_df.groupby("ticker")["Close"].pct_change(1)
        
        # Get rankings from trained model
        service = RankingService(universe=request.universe)
        feature_df = service.engineer_features(stock_df)
        ranks_df = service.rank_market(feature_df, return_scores=False)
        
        # Backtest
        engine = BacktestEngine(
            risk_free_rate=0.04,
            transaction_cost=0.001,
            slippage=0.0005,
        )
        
        if request.strategy_type == "long_only":
            result = engine.backtest_long_only(
                returns_df=stock_df[["date", "ticker", "return"]],
                ranks_df=ranks_df,
                long_n=request.long_n,
            )
        else:
            result = engine.backtest_ranking_strategy(
                returns_df=stock_df[["date", "ticker", "return"]],
                ranks_df=ranks_df,
                long_n=request.long_n,
                short_n=request.short_n,
            )
        
        logger.info(f"Backtest complete. Sharpe: {result['metrics'].get('sharpe', 0):.2f}")

        metrics = result["metrics"].copy()
        strat_returns: pd.Series = result["returns"]

        # ── SPY benchmark ──────────────────────────────────────────────────
        try:
            start_str = "2022-01-01"
            spy_raw = yf.download("SPY", start=start_str, progress=False, auto_adjust=True)
            if not spy_raw.empty:
                spy_close = spy_raw["Close"].squeeze()
                spy_ret = spy_close.pct_change().dropna()
                spy_aligned = spy_ret.reindex(strat_returns.index).fillna(0)
                bh_cum = (1 + spy_aligned).cumprod()
                buy_hold_return = float(bh_cum.iloc[-1] - 1) if len(bh_cum) > 0 else 0.0
            else:
                buy_hold_return = 0.0
                spy_aligned = pd.Series(0.0, index=strat_returns.index)
                bh_cum = pd.Series(1.0, index=strat_returns.index)
        except Exception as spy_err:
            logger.warning(f"SPY benchmark fetch failed: {spy_err}")
            buy_hold_return = 0.0
            spy_aligned = pd.Series(0.0, index=strat_returns.index)
            bh_cum = pd.Series(1.0, index=strat_returns.index)

        # ── Alpha / Sortino / IR ─────────────────────────────────────────
        strategy_return = float(metrics.get("total_return", 0.0))
        metrics["buy_hold_return"] = buy_hold_return
        metrics["alpha"] = strategy_return - buy_hold_return

        rfr_daily = 0.04 / 252
        excess = strat_returns.values - rfr_daily
        down = excess[excess < 0]
        downside_std = float(np.std(down)) if len(down) > 1 else 1e-9
        metrics["sortino"] = float((excess.mean() / downside_std) * np.sqrt(252))

        if len(spy_aligned) == len(strat_returns):
            active = strat_returns.values - spy_aligned.values
            metrics["information_ratio"] = float(
                active.mean() / (active.std() + 1e-9) * np.sqrt(252)
            )

        # ── Equity curve ────────────────────────────────────────────────
        strat_cum = (1 + strat_returns).cumprod() - 1
        equity_curve = []
        for date, strat_val in strat_cum.items():
            bh_val = float(bh_cum.get(date, 1.0)) - 1
            equity_curve.append({
                "date":      str(date)[:10],
                "strategy":  round(float(strat_val), 6),
                "benchmark": round(float(bh_val), 6),
            })

        return BacktestResponse(
            universe=request.universe,
            strategy_type=request.strategy_type,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check for market endpoints."""
    return {"status": "ok", "mode": "market"}
