"""
run_train.py
============
End-to-end script: data → features → train → evaluate → save

Usage:
    python run_train.py --ticker AAPL
    python run_train.py --ticker RELIANCE.NS --start 2018-01-01 --epochs 150
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.pipeline import DataConfig, load_and_prepare
from models.lstm import LSTMForecaster, ModelConfig
from models.train import train
from evaluation.metrics import full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train LSTM stock forecaster")
    parser.add_argument("--ticker",  default="AAPL",       help="Yahoo Finance ticker")
    parser.add_argument("--start",   default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--seq_len", default=60,  type=int, help="Lookback window (trading days)")
    parser.add_argument("--epochs",  default=100, type=int, help="Max training epochs")
    parser.add_argument("--hidden",  default=128, type=int, help="LSTM hidden size")
    parser.add_argument("--layers",  default=2,   type=int, help="Number of LSTM layers")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n{'='*55}")
    print(f" TRAINING: {ticker}")
    print(f"{'='*55}")

    # 1. Load + prepare data
    cfg = DataConfig(ticker=ticker, start=args.start, seq_len=args.seq_len)
    data = load_and_prepare(cfg)

    # 2. Build model
    n_features = data["X_train"].shape[2]
    feature_cols = data.get("feature_cols", [f"feat_{i}" for i in range(n_features)])
    print(f"\nFeatures ({n_features}): {feature_cols}")


    model_cfg = ModelConfig(
        input_size=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=0.3,
        use_attention=True,
    )
    model = LSTMForecaster(model_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training samples: {len(data['X_train']):,}")
    print(f"Param/sample ratio: {total_params / len(data['X_train']):.1f}x")

    # 3. Train
    result = train(
        model, data,
        lr=1e-3,
        epochs=args.epochs,
        batch_size=64,
        patience=15,
        save_dir="checkpoints",
        ticker=ticker,
    )

    # 4. Full evaluation report
    full_evaluation(
        result["y_true"],
        result["y_pred"],
        ticker=ticker,
        verbose=True,
    )

    print(f"\nNext steps:")
    print(f"  1. Start API:     uvicorn api.main:app --reload")
    print(f"  2. Open frontend: cd frontend && npm run dev")
    print(f"  3. Analyze:       GET http://localhost:8000/api/analyze/{ticker}")


if __name__ == "__main__":
    main()