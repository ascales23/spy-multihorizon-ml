"""
Multi-horizon MLP for binary classification.

Architecture sized for ~5K daily samples and ~30 features:
  input → 64 → 32 → N (one logit per horizon)
  Dropout 0.3 between layers, BatchNorm before activations.

~2,500 trainable parameters. Tiny by ML standards, the right size for
the data on hand. If/when you have hourly bars (~500K samples) we'll
swap this for an LSTM or small Transformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiHorizonMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        # One logit per horizon
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))   # logits (no sigmoid, BCEWithLogits)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
