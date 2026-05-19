"""
Market Analysis Service
=======================
Orchestrate the multi-stock ranking engine.

Pipeline:
  1. Load stock universe
  2. Merge macro data
  3. Generate features
  4. Score all stocks
  5. Rank descending
  6. Generate long/short portfolios
  7. Return macro interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from services.market_dataset import MarketDataset, DatasetConfig
from services.macro_service import MacroService, MacroConfig
from services.feature_engineering import FeatureEngineer, FeatureConfig
from models.model_registry import get_registry, get_default_model

logger = logging.getLogger(__name__)


class RankingService:
    """
    End-to-end market ranking engine.
    
    Handles data loading, feature engineering, and model inference.
    """

    def __init__(
        self,
        universe: str = "sp500_tech",
        model_name: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize ranking service.
        
        Args:
            universe: Stock universe ("sp500_tech", "nasdaq_100", "nifty_50")
            model_name: Trained ranker model name (if None, uses default)
            checkpoint_dir: Path to model checkpoints
        """
        self.universe = universe
        self.checkpoint_dir = checkpoint_dir
        self.registry = get_registry(checkpoint_dir)
        
        # Load model
        if model_name is None:
            model_name = get_default_model(mode="market", checkpoint_dir=checkpoint_dir)
        
        if model_name is None:
            logger.warning("No trained market model found. Analysis will use zero scores.")
            self.model = None
        else:
            self.model = self.registry.load_model(model_name)
            logger.info(f"Loaded market model: {model_name}")

        # Initialize data services
        self.dataset_service = MarketDataset(
            DatasetConfig(
                universe=universe,
                start_date="2023-01-01",
                cache_dir=None,
            )
        )
        
        self.macro_service = MacroService(
            MacroConfig(
                start_date="2023-01-01",
                use_gold=True,
                use_yields=True,
                use_vix=True,
                use_dxy=False,
                use_oil=False,
            )
        )
        
        self.feature_engineer = FeatureEngineer(
            FeatureConfig(
                include_technical=True,
                include_volume=True,
                include_gaps=True,
                include_interactions=True,
            )
        )

    def load_market_data(self) -> pd.DataFrame:
        """
        Load and align all stocks in universe.
        
        Returns:
            DataFrame: date | ticker | OHLCV | target_rank
        """
        logger.info(f"Loading market data for {self.universe}...")
        
        # Load all stocks
        stock_data = self.dataset_service.load_all_stocks()
        
        # Align to common dates
        df = self.dataset_service.align_to_common_dates(stock_data)
        
        # Compute cross-sectional targets
        df = self.dataset_service.compute_cross_sectional_target(df, forward_days=5)
        
        logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
        
        return df

    def engineer_features(
        self,
        stock_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate all features (stock + macro interactions).
        
        Args:
            stock_df: Stock OHLCV data
            macro_df: Macro features (if None, loads automatically)
        
        Returns:
            Feature-rich DataFrame
        """
        logger.info("Engineering features...")
        
        # Load macro if needed
        if macro_df is None:
            macro_df = self.macro_service.load_all_macro()
            macro_df = self.macro_service.compute_macro_features(macro_df)
        
        # Engineer stock features
        df = self.feature_engineer.engineer_stock_features(stock_df)
        
        # Add macro interactions
        df = self.feature_engineer.add_macro_interactions(df, macro_df)
        
        # Drop NaN rows
        df = self.feature_engineer.drop_nan_rows(df)
        
        logger.info(f"Features generated. Shape: {df.shape}")
        
        return df

    def rank_market(
        self,
        df: pd.DataFrame,
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """
        Rank all stocks using trained model.
        
        Args:
            df: Feature-engineered DataFrame
            return_scores: If True, return raw scores; False = ranks
        
        Returns:
            DataFrame: date | ticker | rank (or score)
        """
        if self.model is None:
            logger.warning("No model available. Returning zero scores.")
            df_result = df[["date", "ticker"]].copy()
            df_result["rank"] = 1
            return df_result

        feature_cols = self.feature_engineer.get_feature_list(include_target=False)
        valid_cols = [c for c in feature_cols if c in df.columns]
        
        logger.info(f"Ranking with {len(valid_cols)} features...")
        
        ranked = self.model.rank_stocks(
            df[["date", "ticker"] + valid_cols],
            valid_cols,
            return_scores=return_scores,
        )
        
        return ranked

    def get_portfolio(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
        bottom_n: int = 5,
        date: Optional[datetime] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate long and short portfolio recommendations.
        
        Args:
            df: Feature-engineered and ranked DataFrame
            top_n: Number of long candidates
            bottom_n: Number of short candidates
            date: Specific date (if None, uses latest)
        
        Returns:
            {long: [...], short: [...]}
        """
        if date is None:
            date = df["date"].max()
        
        date_df = df[df["date"] == date].copy()
        
        if "rank" not in date_df.columns:
            # If no rank column, compute scores first
            if self.model is None:
                logger.warning("No model to rank with")
                date_df["rank"] = 1
            else:
                feature_cols = self.feature_engineer.get_feature_list(include_target=False)
                valid_cols = [c for c in feature_cols if c in date_df.columns]
                ranked = self.model.rank_stocks(
                    date_df[["date", "ticker"] + valid_cols],
                    valid_cols,
                    return_scores=False,
                )
                date_df = ranked[ranked["date"] == date]
        
        longs = date_df.nsmallest(top_n, "rank")["ticker"].tolist()
        shorts = date_df.nlargest(bottom_n, "rank")["ticker"].tolist()
        
        return {"long": longs, "short": shorts}

    def get_macro_context(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get interpretable macro context for a date.
        
        Args:
            date: Specific date (if None, uses today)
        
        Returns:
            Macro interpretation
        """
        if date is None:
            date = datetime.now()
        
        context = self.macro_service.get_macro_context(date)
        
        return context

    def analyze_market(
        self,
        top_n: int = 5,
        bottom_n: int = 5,
        include_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Full market analysis pipeline.
        
        Returns:
            {
                mode: "market",
                date: "...",
                macro_context: {...},
                long: [...],
                short: [...],
                metrics: {...},
            }
        """
        logger.info("Starting market analysis...")
        
        # Load data
        stock_df = self.load_market_data()
        
        # Engineer features
        feature_df = self.engineer_features(stock_df)
        
        # Rank stocks
        ranked_df = self.rank_market(feature_df, return_scores=False)
        
        # Get portfolio
        portfolio = self.get_portfolio(
            ranked_df,
            top_n=top_n,
            bottom_n=bottom_n,
        )
        
        # Get macro context
        macro_context = self.get_macro_context()
        
        # Compute metrics if requested
        metrics = {}
        if include_metrics and self.model is not None:
            try:
                feature_cols = self.feature_engineer.get_feature_list(include_target=False)
                valid_cols = [c for c in feature_cols if c in feature_df.columns]
                
                rank_ic = self.model.eval_rank_ic(
                    feature_df[["date", "ticker"] + valid_cols + ["target_rank"]],
                    valid_cols,
                )
                metrics["rank_ic"] = float(rank_ic)
                
                feature_imp = self.model.get_feature_importance(top_n=5)
                metrics["top_features"] = feature_imp
            except Exception as e:
                logger.warning(f"Could not compute metrics: {e}")

        result = {
            "mode": "market",
            "date": str(datetime.now()),
            "macro_context": macro_context,
            "long": portfolio["long"],
            "short": portfolio["short"],
            "metrics": metrics,
        }
        
        logger.info(f"Analysis complete. Long: {len(result['long'])}, Short: {len(result['short'])}")
        
        return result

    def get_stock_details(
        self,
        ticker: str,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed analysis for a single stock.
        
        Args:
            ticker: Stock symbol
            date: Specific date (if None, uses latest)
        
        Returns:
            Stock details and features
        """
        # Load data
        stock_df = self.load_market_data()
        
        # Engineer features
        feature_df = self.engineer_features(stock_df)
        
        # Filter to ticker
        ticker_df = feature_df[feature_df["ticker"] == ticker]
        
        if ticker_df.empty:
            return {"error": f"No data for {ticker}"}
        
        if date is None:
            date = ticker_df["date"].max()
        
        ticker_date_df = ticker_df[ticker_df["date"] == date]
        
        if ticker_date_df.empty:
            return {"error": f"No data for {ticker} on {date}"}
        
        row = ticker_date_df.iloc[0]
        
        # Get rank
        rank = None
        if self.model is not None:
            feature_cols = self.feature_engineer.get_feature_list(include_target=False)
            valid_cols = [c for c in feature_cols if c in feature_df.columns]
            
            # Rank all on this date
            date_df = feature_df[feature_df["date"] == date]
            ranked = self.model.rank_stocks(
                date_df[["date", "ticker"] + valid_cols],
                valid_cols,
                return_scores=False,
            )
            ticker_ranked = ranked[ranked["ticker"] == ticker]
            if not ticker_ranked.empty:
                rank = ticker_ranked.iloc[0]["rank"]
        
        result = {
            "ticker": ticker,
            "date": str(date),
            "rank": rank,
            "features": row.to_dict(),
        }
        
        return result
