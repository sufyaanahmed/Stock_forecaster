"""
Training Script for Market Mode
===============================
Train the new macro-aware multi-stock LambdaRank ranking model.

This demonstrates:
  1. Loading multi-stock data
  2. Computing macro features
  3. Engineering cross-sectional features
  4. Training LambdaRank model
  5. Evaluating with Rank IC
  6. Registering the model
"""

import sys
import logging
from pathlib import Path
import argparse
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from services.market_dataset import MarketDataset, DatasetConfig
from services.macro_service import MacroService, MacroConfig
from services.feature_engineering import FeatureEngineer, FeatureConfig
from models.ranker import RankerModel
from models.model_registry import register_model


def main():
    parser = argparse.ArgumentParser(description="Train market ranking model")
    parser.add_argument("--universe", default="sp500_tech", help="Stock universe")
    parser.add_argument("--model-name", default="market_v2.0", help="Model name for registry")
    parser.add_argument("--num-rounds", type=int, default=100, help="LightGBM boost rounds")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MARKET MODE TRAINING")
    logger.info("=" * 80)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Load multi-stock data
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info(f"Loading {args.universe} universe...")
    dataset = MarketDataset(
        DatasetConfig(
            universe=args.universe,
            start_date=args.start_date,
        )
    )
    stock_df = dataset.build_dataset()
    
    if stock_df.empty:
        logger.error("No data loaded!")
        return False
    
    logger.info(f"Loaded {len(stock_df)} rows, {stock_df['ticker'].nunique()} tickers")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 2. Load and compute macro features
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info("Loading macro data...")
    macro_service = MacroService(
        MacroConfig(
            start_date=args.start_date,
            use_gold=True,
            use_yields=True,
            use_vix=True,
            use_dxy=False,
            use_oil=False,
        )
    )
    macro_df = macro_service.load_all_macro()
    macro_df = macro_service.compute_macro_features(macro_df)
    
    logger.info(f"Loaded macro data: {len(macro_df)} dates, {len(macro_df.columns)} features")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 3. Engineer features
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info("Engineering features...")
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
    
    logger.info(f"Generated features. Shape: {feature_df.shape}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 4. Prepare data for training
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info("Preparing training data...")
    feature_cols = engineer.get_feature_list(include_target=False)
    valid_cols = [c for c in feature_cols if c in feature_df.columns]
    
    logger.info(f"Using {len(valid_cols)} features:")
    for i, col in enumerate(valid_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Prepare LambdaRank data
    X, y, group_sizes, dates = (
        feature_df[valid_cols].fillna(0).values.astype(np.float32),
        feature_df["target_rank"].fillna(0.5).values.astype(np.float32),
        feature_df.groupby("date").size().values,
        feature_df["date"].values,
    )
    
    logger.info(f"Data prepared:")
    logger.info(f"  X: {X.shape}")
    logger.info(f"  y: {y.shape}")
    logger.info(f"  Groups: {len(group_sizes)}, avg size: {group_sizes.mean():.1f}")
    logger.info(f"  Date range: {dates.min()} to {dates.max()}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 5. Train model
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info(f"Training LambdaRank model ({args.num_rounds} rounds)...")
    model = RankerModel()
    
    trained_model = model.train(
        X=X,
        y=y,
        group_sizes=group_sizes,
        dates=dates,
        feature_names=valid_cols,
        num_boost_round=args.num_rounds,
        train_ratio=0.80,
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # 6. Evaluate model
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info("Evaluating model...")
    
    # Rank IC
    rank_ic = model.eval_rank_ic(
        feature_df[["date", "ticker"] + valid_cols + ["target_rank"]],
        valid_cols,
    )
    
    # Feature importance
    feature_imp = model.get_feature_importance(top_n=10)
    logger.info("Top 10 features by importance:")
    for i, (feat, imp) in enumerate(feature_imp.items(), 1):
        print(f"  {i:2d}. {feat:30s}: {imp:8.2f}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 7. Register model
    # ──────────────────────────────────────────────────────────────────────────
    
    logger.info(f"Registering model as '{args.model_name}'...")
    
    metrics = {
        "rank_ic": float(rank_ic),
        "num_samples": len(feature_df),
        "num_features": len(valid_cols),
        "num_boost_rounds": args.num_rounds,
        "universe": args.universe,
        "date_range": f"{dates.min()} to {dates.max()}",
    }
    
    register_model(
        model_name=args.model_name,
        model_obj=model,
        mode="market",
        version="2.0",
        metrics=metrics,
    )
    
    logger.info("=" * 80)
    logger.info("✓ Training complete!")
    logger.info(f"✓ Model registered: {args.model_name}")
    logger.info(f"✓ Rank IC: {rank_ic:.4f}")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info(f"  1. Test the model with: curl http://localhost:8000/market/analyze")
    logger.info(f"  2. Backtest strategy with: curl -X POST http://localhost:8000/market/backtest")
    logger.info(f"  3. Train another model or adjust hyperparameters")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
