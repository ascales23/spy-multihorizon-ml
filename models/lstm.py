"""
Multi-scale LSTM for multi-horizon ternary classification.

Architecture
------------
Three parallel branches, one per temporal window:

  1D branch (current bar):
      Linear(n_features → d_model)  →  ReLU  →  d_model vector

  1W branch (last 5 bars):
      LSTM(n_features, hidden_1w, num_layers=1)  →  last hidden state

  1M branch (last 21 bars):
      LSTM(n_features, hidden_1m, num_layers=2, dropout=0.2)  →  last hidden

Fusion:
      concat(d_model, hidden_1w, hidden_1m)
      → LayerNorm → Dropout(p)
      → Linear(fused → d_fc) → ReLU → Dropout(p)
      → one head per horizon, each outputting 3 logits (short / abstain / long)

Loss: CrossEntropyLoss per horizon, summed. Optional class weights to
handle the abstain-heavy class imbalance from SR-gated labeling.

~100K–200K parameters depending on hidden sizes, right-sized for
daily data with ~1500 samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_classes: int = 3,           # 0=short, 1=abstain, 2=long
        d_model: int = 64,            # 1D branch embedding size
        hidden_1w: int = 64,          # 1W LSTM hidden size
        hidden_1m: int = 128,         # 1M LSTM hidden size
        lstm_1m_layers: int = 2,
        d_fc: int = 128,              # fusion FC size
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_classes  = n_classes

        # 1D branch: single-bar snapshot → embedding
        self.branch_1d = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 1W branch: 5-bar sequence
        self.lstm_1w = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_1w,
            num_layers=1,
            batch_first=True,
        )

        # 1M branch: 21-bar sequence (deeper)
        self.lstm_1m = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_1m,
            num_layers=lstm_1m_layers,
            batch_first=True,
            dropout=dropout if lstm_1m_layers > 1 else 0.0,
        )

        fused_size = d_model + hidden_1w + hidden_1m
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_size),
            nn.Dropout(dropout),
            nn.Linear(fused_size, d_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # One classification head per horizon
        self.heads = nn.ModuleList([
            nn.Linear(d_fc, n_classes) for _ in range(n_horizons)
        ])

    def forward(
        self,
        x_1d: torch.Tensor,   # (B, F)
        x_1w: torch.Tensor,   # (B, 5, F)
        x_1m: torch.Tensor,   # (B, 21, F)
    ) -> list[torch.Tensor]:
        """Returns a list of n_horizons logit tensors, each (B, n_classes)."""
        h_1d = self.branch_1d(x_1d)                    # (B, d_model)

        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]                                 # (B, hidden_1w)

        _, (h_1m, _) = self.lstm_1m(x_1m)
        h_1m = h_1m[-1]                                 # (B, hidden_1m)

        fused = self.fusion(torch.cat([h_1d, h_1w, h_1m], dim=1))  # (B, d_fc)
        return [head(fused) for head in self.heads]     # list of (B, 3)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_proba(
        self,
        x_1d: torch.Tensor,
        x_1w: torch.Tensor,
        x_1m: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Softmax probabilities per horizon. (B, 3) each."""
        logits = self.forward(x_1d, x_1w, x_1m)
        return [torch.softmax(l, dim=-1) for l in logits]
