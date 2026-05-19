"""
Market System Configuration
===========================
Central configuration for the new macro-aware multi-stock ranking engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MarketConfig:
    """Central configuration for market mode."""
    
    # Data & Universe
    DEFAULT_UNIVERSE: str = "sp500_tech"
    AVAILABLE_UNIVERSES: Dict[str, List[str]] = None
    
    # Feature Engineering
    INCLUDE_TECHNICAL: bool = True
    INCLUDE_VOLUME: bool = True
    INCLUDE_GAPS: bool = True
    INCLUDE_INTERACTIONS: bool = True
    LOOKBACK_WINDOW: int = 60
    
    # Macro Data
    USE_GOLD: bool = True
    USE_YIELDS: bool = True
    USE_FED_RATE: bool = False
    USE_VIX: bool = True
    USE_DXY: bool = False
    USE_OIL: bool = False
    
    # Model Training
    RANKER_OBJECTIVE: str = "lambdarank"
    RANKER_NUM_LEAVES: int = 31
    RANKER_LEARNING_RATE: float = 0.1
    RANKER_NUM_BOOST_ROUNDS: int = 100
    TRAIN_RATIO: float = 0.80
    
    # Portfolio Construction
    DEFAULT_LONG_N: int = 5
    DEFAULT_SHORT_N: int = 5
    
    # Backtest
    RISK_FREE_RATE: float = 0.04  # 4% annual
    TRANSACTION_COST: float = 0.001  # 0.1%
    SLIPPAGE: float = 0.0005  # 0.05%
    
    # Caching
    DATA_CACHE_TTL: int = 3600  # 1 hour
    MACRO_CACHE_TTL: int = 3600
    CHART_CACHE_TTL: int = 300  # 5 minutes
    
    # API
    DEFAULT_TOP_N: int = 5
    DEFAULT_BOTTOM_N: int = 5
    MAX_TOP_N: int = 20
    MAX_BOTTOM_N: int = 20
    
    def __post_init__(self):
        """Initialize default universes if not provided."""
        if self.AVAILABLE_UNIVERSES is None:
            self.AVAILABLE_UNIVERSES = {
                "sp500_tech": ["NVDA", "META", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX"],
                "nasdaq_100": ["NVDA", "META", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX", "ADBE", "CRM"],
                "nifty_50": ["INFY", "TCS", "RELIANCE", "HINDUNILVR", "ICICIBANK"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MARKET_CONFIG = MarketConfig()


def get_market_config() -> MarketConfig:
    """Get global market configuration."""
    return DEFAULT_MARKET_CONFIG


def set_market_config(config: MarketConfig):
    """Set global market configuration."""
    global DEFAULT_MARKET_CONFIG
    DEFAULT_MARKET_CONFIG = config
