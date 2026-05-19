"""
Model Registry
==============
Runtime switching between legacy (LSTM) and market (LambdaRank) modes.

Allows:
  - load_model("legacy") → Single-stock LSTM
  - load_model("market") → Multi-stock LambdaRank ranker
  - list_models() → Available trained models
  - register_model() → Add new models

This enables A/B testing and smooth transitions.
"""

import logging
from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for managing multiple models.
    
    Supports both legacy and market modes.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._models: Dict[str, Any] = {}  # In-memory cache
        self._metadata: Dict[str, Dict] = {}  # Model metadata
        
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata about all registered models."""
        metadata_file = self.checkpoint_dir / "registry.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self._metadata = json.load(f)
            logger.info(f"Loaded registry with {len(self._metadata)} models")
        else:
            self._metadata = {}

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_file = self.checkpoint_dir / "registry.json"
        with open(metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def register_model(
        self,
        model_name: str,
        model_obj: Any,
        mode: Literal["legacy", "market"],
        version: str = "1.0",
        metrics: Optional[Dict] = None,
    ):
        """
        Register a trained model.
        
        Args:
            model_name: Unique model identifier
            model_obj: Model object (LSTM or Ranker)
            mode: "legacy" or "market"
            version: Model version
            metrics: Training metrics
        """
        self._models[model_name] = model_obj
        
        self._metadata[model_name] = {
            "mode": mode,
            "version": version,
            "metrics": metrics or {},
            "checkpoint": f"{model_name}.pkl",
        }
        
        # Save model to disk
        checkpoint_path = self.checkpoint_dir / f"{model_name}.pkl"
        
        # Handle different model types
        if hasattr(model_obj, "save"):
            model_obj.save(str(checkpoint_path))
        else:
            # Fallback: pickle
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model_obj, f)
        
        self._save_metadata()
        logger.info(f"Registered {mode} model: {model_name} v{version}")

    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """
        Load a model by name.
        
        Args:
            model_name: Model identifier
            force_reload: If True, reload from disk even if cached
        
        Returns:
            Model object (LSTM or Ranker)
        """
        # Check cache
        if model_name in self._models and not force_reload:
            logger.info(f"Loaded {model_name} from cache")
            return self._models[model_name]

        # Load from disk
        if model_name not in self._metadata:
            raise ValueError(f"Unknown model: {model_name}")

        checkpoint_path = self.checkpoint_dir / self._metadata[model_name]["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Try loading with .load() method first (LambdaRank)
            model_obj = self._create_model_instance(self._metadata[model_name]["mode"])
            if hasattr(model_obj, "load"):
                model_obj.load(str(checkpoint_path))
            else:
                # Fallback: unpickle
                with open(checkpoint_path, "rb") as f:
                    model_obj = pickle.load(f)
            
            self._models[model_name] = model_obj
            logger.info(f"Loaded {model_name} from disk")
            return model_obj

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def _create_model_instance(self, mode: str) -> Any:
        """Create empty model instance for loading."""
        if mode == "legacy":
            from models.lstm import LSTMForecaster, ModelConfig
            return LSTMForecaster(ModelConfig(input_size=10))  # Placeholder
        elif mode == "market":
            from models.ranker import RankerModel
            return RankerModel()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def list_models(self, mode: Optional[str] = None) -> Dict[str, Dict]:
        """
        List all registered models.
        
        Args:
            mode: Filter by "legacy" or "market" (None = all)
        
        Returns:
            {model_name: {mode, version, metrics}}
        """
        if mode is None:
            return self._metadata
        else:
            return {
                name: meta for name, meta in self._metadata.items()
                if meta.get("mode") == mode
            }

    def get_default_model(self, mode: Literal["legacy", "market"] = "market") -> Optional[str]:
        """
        Get the default model for a mode.
        
        Convention: highest version is default.
        
        Args:
            mode: "legacy" or "market"
        
        Returns:
            Model name or None
        """
        models_by_mode = self.list_models(mode=mode)
        
        if not models_by_mode:
            return None

        # Sort by version (higher = newer)
        sorted_models = sorted(
            models_by_mode.items(),
            key=lambda x: self._version_to_tuple(x[1].get("version", "0.0")),
            reverse=True
        )

        return sorted_models[0][0]

    @staticmethod
    def _version_to_tuple(version_str: str) -> tuple:
        """Convert version string to tuple for comparison."""
        return tuple(map(int, version_str.split(".")))

    def unload_model(self, model_name: str):
        """Remove model from memory cache."""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded {model_name}")

    def get_model_metadata(self, model_name: str) -> Dict:
        """Get metadata for a model."""
        if model_name not in self._metadata:
            raise ValueError(f"Unknown model: {model_name}")
        return self._metadata[model_name]

    def update_metrics(self, model_name: str, metrics: Dict):
        """Update metrics for a model."""
        if model_name not in self._metadata:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._metadata[model_name]["metrics"].update(metrics)
        self._save_metadata()
        logger.info(f"Updated metrics for {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_global_registry: Optional[ModelRegistry] = None


def get_registry(checkpoint_dir: str = "checkpoints") -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry(checkpoint_dir)
    return _global_registry


def load_model(model_name: str, checkpoint_dir: str = "checkpoints") -> Any:
    """Convenience function to load a model."""
    registry = get_registry(checkpoint_dir)
    return registry.load_model(model_name)


def register_model(
    model_name: str,
    model_obj: Any,
    mode: Literal["legacy", "market"],
    version: str = "1.0",
    metrics: Optional[Dict] = None,
    checkpoint_dir: str = "checkpoints",
):
    """Convenience function to register a model."""
    registry = get_registry(checkpoint_dir)
    registry.register_model(model_name, model_obj, mode, version, metrics)


def list_models(mode: Optional[str] = None, checkpoint_dir: str = "checkpoints") -> Dict:
    """Convenience function to list models."""
    registry = get_registry(checkpoint_dir)
    return registry.list_models(mode=mode)


def get_default_model(
    mode: Literal["legacy", "market"] = "market",
    checkpoint_dir: str = "checkpoints"
) -> Optional[str]:
    """Convenience function to get default model."""
    registry = get_registry(checkpoint_dir)
    return registry.get_default_model(mode)
