"""
LightGBM LambdaRank Model
========================
Cross-sectional ranking model using LambdaRank objective.

This is NOT a regression model predicting absolute prices.
It's a ranking model that learns: "Which stocks outperform peers?"

Key differences:
  - Objective: lambdarank (not MSE)
  - Grouped by trading day (all stocks on same day form a group)
  - Metric: NDCG (Normalized Discounted Cumulative Gain)
  - Output: Ranking scores (not prices)
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class RankerModel:
    """
    LightGBM LambdaRank for cross-sectional stock ranking.
    """

    DEFAULT_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "num_threads": 4,
        "verbose": -1,
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importances = None

    def prepare_ranking_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_rank",
    ) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
        """
        Prepare data for LightGBM LambdaRank.
        
        LambdaRank requires:
          - Features (X)
          - Targets (y) — relevance scores
          - Group sizes — how many items per group (date)
        
        Args:
            df: DataFrame with date, ticker, features, target
            feature_cols: Column names of features
            target_col: Name of target column (relevance/percentile rank)
        
        Returns:
            (X, y, group_sizes, dates)
        """
        # Ensure data is sorted by date and ticker
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        # Extract features and targets
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        y = df[target_col].fillna(0.5).values.astype(np.float32)
        
        # Group sizes: how many stocks per date
        group_sizes = df.groupby("date").size().values
        
        # Keep dates for reference
        dates = df["date"].values
        
        logger.info(f"Prepared LambdaRank data:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Groups: {len(group_sizes)}, avg size: {group_sizes.mean():.1f}")
        
        return X, y, group_sizes, dates

    def split_train_val(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: List[int],
        dates: np.ndarray,
        train_ratio: float = 0.8,
        chronological: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, List[int]], Tuple[np.ndarray, np.ndarray, List[int]]]:
        """
        Split data into train/val chronologically.
        
        IMPORTANT: For trading models, NEVER random shuffle!
        Always use chronological splits (walk-forward).
        
        Args:
            X, y, group_sizes, dates: From prepare_ranking_data
            train_ratio: Fraction for training
            chronological: If True, split by date (recommended for trading)
        
        Returns:
            ((X_train, y_train, groups_train), (X_val, y_val, groups_val))
        """
        if chronological:
            # Find split point by date
            unique_dates = np.unique(dates)
            split_idx = int(len(unique_dates) * train_ratio)
            split_date = unique_dates[split_idx]
            
            train_mask = dates < split_date
            val_mask = dates >= split_date
            
            logger.info(f"Chronological split: {train_mask.sum()} train, {val_mask.sum()} val")
            logger.info(f"  Train dates: {dates[train_mask].min()} to {dates[train_mask].max()}")
            logger.info(f"  Val dates: {dates[val_mask].min()} to {dates[val_mask].max()}")
        else:
            split_idx = int(len(X) * train_ratio)
            train_mask = np.arange(len(X)) < split_idx
            val_mask = ~train_mask

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        
        # Recompute group sizes for train/val
        train_dates = dates[train_mask]
        val_dates = dates[val_mask]
        
        groups_train = []
        for d in np.unique(train_dates):
            groups_train.append((train_dates == d).sum())
        
        groups_val = []
        for d in np.unique(val_dates):
            groups_val.append((val_dates == d).sum())
        
        return (X_train, y_train, groups_train), (X_val, y_val, groups_val)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: List[int],
        dates: np.ndarray,
        feature_names: List[str],
        num_boost_round: int = 100,
        train_ratio: float = 0.8,
    ):
        """
        Train LambdaRank model.
        
        Args:
            X, y, group_sizes, dates: Prepared ranking data
            feature_names: Names of features
            num_boost_round: Number of boosting iterations
            train_ratio: Train/val split ratio
        """
        self.feature_names = feature_names
        
        # Split chronologically
        (X_train, y_train, groups_train), (X_val, y_val, groups_val) = self.split_train_val(
            X, y, group_sizes, dates, train_ratio=train_ratio, chronological=True
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            feature_names=feature_names,
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            group=groups_val,
            feature_names=feature_names,
            reference=train_data,
        )
        
        # Train
        logger.info("Training LambdaRank model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.log_evaluation(period=20),
                lgb.early_stopping(stopping_rounds=10),
            ],
        )
        
        # Feature importance
        self.feature_importances = self.model.feature_importance(importance_type="gain")
        
        logger.info("Training complete")
        
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Score stocks (ranking scores).
        
        Args:
            X: Feature matrix
        
        Returns:
            Ranking scores (higher = better)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        scores = self.model.predict(X)
        return scores

    def rank_stocks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """
        Rank stocks using trained model.
        
        Args:
            df: DataFrame with date, ticker, features
            feature_cols: Column names of features
            return_scores: If True, return raw scores; if False, return ranks
        
        Returns:
            DataFrame with date, ticker, rank (or score)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        X = df[feature_cols].fillna(0).values
        scores = self.predict(X)
        
        df_result = df[["date", "ticker"]].copy()
        df_result["score"] = scores
        
        if not return_scores:
            # Rank within each date (1 = best, n = worst)
            df_result["rank"] = df_result.groupby("date")["score"].rank(ascending=False)
        
        return df_result

    def get_top_stocks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        top_n: int = 5,
        bottom_n: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Get top long and short candidates.
        
        Args:
            df: DataFrame with features
            feature_cols: Feature columns
            top_n: Number of long candidates
            bottom_n: Number of short candidates
        
        Returns:
            {long: [tickers], short: [tickers]}
        """
        ranked = self.rank_stocks(df, feature_cols, return_scores=False)
        
        # Get latest date
        latest_date = ranked["date"].max()
        latest = ranked[ranked["date"] == latest_date]
        
        longs = latest.nsmallest(top_n, "rank")["ticker"].tolist()
        shorts = latest.nlargest(bottom_n, "rank")["ticker"].tolist()
        
        return {"long": longs, "short": shorts}

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Return top important features."""
        if self.feature_importances is None:
            return {}
        
        importance_dict = dict(zip(self.feature_names, self.feature_importances))
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return dict(top_features)

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "params": self.params,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.model = state["model"]
        self.feature_names = state["feature_names"]
        self.feature_importances = state["feature_importances"]
        self.params = state["params"]
        
        logger.info(f"Loaded model from {path}")

    def eval_ndcg(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_rank",
    ) -> float:
        """
        Evaluate NDCG@5 on test set.
        
        NDCG measures ranking quality.
        Perfect ranking = 1.0
        Random ranking ≈ 0.5
        """
        X, y, group_sizes, _ = self.prepare_ranking_data(df, feature_cols, target_col)
        scores = self.predict(X)
        
        # Compute NDCG per group (date)
        ndcgs = []
        idx = 0
        for group_size in group_sizes:
            group_scores = scores[idx : idx + group_size]
            group_targets = y[idx : idx + group_size]
            
            # Sort by predicted score
            sorted_indices = np.argsort(-group_scores)
            sorted_targets = group_targets[sorted_indices]
            
            # NDCG@5
            ideal_targets = np.sort(group_targets)[::-1]
            
            def dcg(targets, k=5):
                targets = targets[:k]
                if len(targets) == 0:
                    return 0.0
                gains = 2 ** targets - 1
                discounts = np.log2(np.arange(len(targets)) + 2)
                return np.sum(gains / discounts)
            
            dcg_val = dcg(sorted_targets, k=5)
            ideal_dcg = dcg(ideal_targets, k=5)
            
            ndcg = dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)
            
            idx += group_size
        
        mean_ndcg = np.mean(ndcgs)
        logger.info(f"NDCG@5: {mean_ndcg:.4f}")
        
        return mean_ndcg

    def eval_rank_ic(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_rank",
    ) -> float:
        """
        Rank Information Coefficient.
        
        Spearman correlation between predicted rank and actual rank.
        Measures predictive power: 0 = useless, 1 = perfect.
        """
        from scipy.stats import spearmanr
        
        X, y, _, _ = self.prepare_ranking_data(df, feature_cols, target_col)
        scores = self.predict(X)
        
        # Spearman correlation
        ic, p_value = spearmanr(scores, y)
        
        logger.info(f"Rank IC: {ic:.4f} (p-value: {p_value:.4e})")
        
        return float(ic) if not np.isnan(ic) else 0.0
