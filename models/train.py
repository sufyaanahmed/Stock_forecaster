"""
Training Loop
=============
Professional-grade training with:
  - Walk-forward awareness
  - Information Coefficient (IC) as primary metric (not MSE)
  - Early stopping on validation IC
  - Learning rate scheduling
  - Gradient clipping (critical for RNNs)
  - Full experiment logging
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import scipy.stats as stats
from typing import Dict, List
import json, os, time
from pathlib import Path

from models.lstm import LSTMForecaster, ModelConfig


# ── Metrics ──────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and actual returns.
    
    Why IC, not MSE?
    MSE measures magnitude error — a model that predicts ±0.0001 always
    can have low MSE while being directionally useless. In trading,
    DIRECTION matters more than magnitude. You make money if you know
    which way the stock moves, not exactly how much.
    
    IC of 0.02–0.05 is considered meaningful in practice.
    IC > 0.10 is excellent (most quant strategies run on IC ~0.03).
    IC < 0 means your model is anti-predictive (useful if you flip the signal).
    """
    ic, p_value = stats.spearmanr(y_pred, y_true)
    return float(ic) if not np.isnan(ic) else 0.0


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of days where predicted direction matches actual.
    Random baseline: 0.50. Meaningful threshold: > 0.52 consistently.
    """
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "ic":       information_coefficient(y_true, y_pred),
        "dir_acc":  direction_accuracy(y_true, y_pred),
        "mse":      float(np.mean((y_true - y_pred) ** 2)),
        "mae":      float(np.mean(np.abs(y_true - y_pred))),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataloaders(data: dict, batch_size: int = 64) -> tuple:
    def to_loader(X, y, shuffle):
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).unsqueeze(-1)   # (N,) → (N,1)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader   = to_loader(data["X_val"],   data["y_val"],   shuffle=False)
    test_loader  = to_loader(data["X_test"],  data["y_test"],  shuffle=False)
    return train_loader, val_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_ic   = -np.inf
        self.counter   = 0
        self.best_state = None

    def step(self, ic: float, model: nn.Module) -> bool:
        if ic > self.best_ic + self.min_delta:
            self.best_ic    = ic
            self.counter    = 0
            # Save best model weights in memory (not disk on every epoch)
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience   # True → stop


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = nn.MSELoss()(pred, y_batch)
        loss.backward()

        # Gradient clipping — essential for RNNs.
        # Without this, gradients can explode through time, causing NaN weights.
        # max_norm=1.0 is standard; clip L2 norm of all parameter gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device) -> tuple:
    """Returns (metrics_dict, y_true, y_pred)"""
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_pred.extend(pred.flatten())
            all_true.extend(y_batch.numpy().flatten())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    return compute_metrics(y_true, y_pred), y_true, y_pred


def train(
    model: LSTMForecaster,
    data: dict,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    save_dir: str = "checkpoints",
    ticker: str = "UNKNOWN",
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    train_loader, val_loader, test_loader = make_dataloaders(data, batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # CosineAnnealingLR: smoothly decays lr from lr → 0 over T_max epochs
    # Better than StepLR for financial data where regime changes require adaptation
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    stopper   = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_ic": [], "val_dir_acc": [], "lr": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val IC':>10} {'Val DirAcc':>12} {'LR':>10}")
    print("─" * 56)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_ic"].append(val_metrics["ic"])
        history["val_dir_acc"].append(val_metrics["dir_acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_loss:>12.5f} {val_metrics['ic']:>10.4f} "
                  f"{val_metrics['dir_acc']:>12.4f} {scheduler.get_last_lr()[0]:>10.6f}")

        if stopper.step(val_metrics["ic"], model):
            print(f"\nEarly stopping at epoch {epoch}. Best Val IC: {stopper.best_ic:.4f}")
            break

    # Restore best weights
    if stopper.best_state:
        model.load_state_dict(stopper.best_state)

    # Final test evaluation
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS [{ticker}]")
    print(f"  IC (Spearman):      {test_metrics['ic']:.4f}")
    print(f"  Direction Accuracy: {test_metrics['dir_acc']:.4f}  (random baseline: 0.50)")
    print(f"  MSE:                {test_metrics['mse']:.6f}")
    print(f"  MAE:                {test_metrics['mae']:.6f}")
    print(f"{'='*50}")

    # Save model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_dir}/{ticker}_lstm.pt"
    torch.save({
        "model_state": stopper.best_state or model.state_dict(),
        "model_config": model.cfg,
        "history": history,
        "test_metrics": test_metrics,
        "feature_cols": data["feature_cols"],
    }, save_path)
    print(f"\nModel saved → {save_path}")

    return {
        "history": history,
        "test_metrics": test_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "save_path": save_path,
    }