"""
PyTorch LSTM Model
==================
Architecture decisions explained mathematically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    input_size: int        # number of features (set from data pipeline)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False  # discussed below
    use_attention: bool = True   # lightweight self-attention on top of LSTM


class LSTMForecaster(nn.Module):
    """
    LSTM with optional attention for financial return forecasting.

    Architecture rationale:
    ──────────────────────
    • 2 layers, not 4: with only ~1500-2000 training samples (sequences),
      deep stacking causes overfitting. 2 layers is a well-validated depth
      for financial time-series. Add layers only if val loss improves.

    • hidden_size=128: enough representational capacity without over-parameterizing.
      Parameter count: ~128k. Sample count: ~1500. Ratio ~85x — acceptable for
      regularized RNNs. The original model's 4-layer architecture had a worse ratio.

    • Dropout between LSTM layers: nn.LSTM has a built-in `dropout` param
      that applies between layers (NOT after the last layer). We add an
      explicit dropout after the final LSTM output for the attention/dense path.

    • Why tanh activation (default in nn.LSTM)?
      LSTM cell uses tanh for cell state update and output squashing:
        C̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)
        h_t = o_t ⊙ tanh(C_t)
      tanh maps to [-1, 1], providing implicit normalization and bounded
      gradients. ReLU is unbounded → exploding gradients in recurrent paths.

    • Bidirectional: False by default for forecasting.
      A bidirectional LSTM reads sequence in both directions, using future
      context. This is valid for NLP (sentence classification uses full context)
      but creates look-ahead bias in forecasting — the model sees future
      timesteps within the sequence window. Only valid if you're not doing
      true one-step-ahead prediction.

    • Attention: lightweight additive attention over LSTM hidden states.
      Allows the model to weight which past timesteps matter most rather
      than relying solely on the final hidden state. Improves interpretability.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: project raw features into model dimension
        # This allows decoupling feature dimensionality from hidden size
        self.input_proj = nn.Linear(cfg.input_size, cfg.hidden_size)
        self.input_norm = nn.LayerNorm(cfg.hidden_size)

        # Core LSTM
        # batch_first=True: input shape (batch, seq, features) — more intuitive
        # dropout applies between LSTM layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional
        )

        lstm_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # Attention mechanism (additive / Bahdanau-style, simplified)
        self.use_attention = cfg.use_attention
        if cfg.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Output head
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(lstm_out_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, 1) — predicted next-day log return
        """
        # Project + normalize input
        x = self.input_proj(x)          # (B, T, hidden)
        x = self.input_norm(x)
        x = F.gelu(x)                   # GELU: smoother than ReLU, used in Transformers

        # LSTM forward pass
        # lstm_out: (B, T, hidden) — all hidden states
        # (h_n, c_n): final hidden and cell state
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.use_attention:
            # Compute attention weights over all timesteps
            # scores: (B, T, 1) → softmax over T → weighted sum
            scores = self.attention(lstm_out)           # (B, T, 1)
            weights = torch.softmax(scores, dim=1)      # (B, T, 1)
            context = (lstm_out * weights).sum(dim=1)   # (B, hidden)
        else:
            # Fallback: use only the last hidden state
            context = lstm_out[:, -1, :]                # (B, hidden)

        # Output head
        out = self.dropout(context)
        out = F.gelu(self.fc1(out))
        out = self.fc2(out)             # (B, 1) — raw return prediction (no activation)
        return out

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Return attention weights for interpretability visualization.
        Shows which past days the model focuses on for each prediction.
        """
        self.eval()
        with torch.no_grad():
            x_proj = F.gelu(self.input_norm(self.input_proj(x)))
            lstm_out, _ = self.lstm(x_proj)
            scores = self.attention(lstm_out)
            weights = torch.softmax(scores, dim=1)
        return weights.squeeze(-1).cpu().numpy()


class NaiveBaseline:
    """
    Persistence model: predicts tomorrow's return = today's return.
    This is your minimum bar to beat.
    If your LSTM can't beat this, it has learned nothing.
    """
    def predict(self, y: np.ndarray) -> np.ndarray:
        return y[:-1]   # shift by one

    def direction_accuracy(self, y_true: np.ndarray) -> float:
        pred = self.predict(y_true)
        actual = y_true[1:]
        return np.mean(np.sign(pred) == np.sign(actual))