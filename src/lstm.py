"""
LSTM model for sequence-based implied volatility prediction.

Architecture: stacked LSTM → Linear output.
Mirrors the training interface of the FC colab utilities for consistency.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Stacked LSTM for d_iv prediction.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_size : int
        Hidden state dimension for each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only applied if num_layers > 1).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2,
                 dropout=0.1, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

        # Initialize output layer
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        # Use last timestep output
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last)  # (batch, 1)
