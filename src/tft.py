"""
Simplified Temporal Fusion Transformer (TFT) for d_iv prediction.

Pure PyTorch implementation realistic for Colab — no pytorch-forecasting
dependency. Implements the key TFT components:

  1. Variable Selection Network (VSN) — learns input feature importance
  2. Gated Residual Network (GRN) — nonlinear feature processing
  3. Multi-head attention — temporal self-attention over the encoder window
  4. Gated skip connections — controls information flow

Uses LOOKBACK=20 as encoder length for fair comparison with LSTM/GRU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    """GLU activation: output = sigmoid(Wx+b) * (Vx+c)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.fc(x)


class GatedResidualNetwork(nn.Module):
    """
    GRN with optional context vector.

    GRN(a, c) = LayerNorm(a + GLU(η₁))
    η₁ = W₁·η₂ + b₁
    η₂ = ELU(W₂·a + W₃·c + b₂)  (c is optional)
    """

    def __init__(self, in_dim, hidden_dim, out_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.glu = GatedLinearUnit(out_dim, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x

        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)

        return self.layernorm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns feature importance weights.

    For each timestep, produces softmax weights over input features and
    applies them to GRN-processed feature representations.
    """

    def __init__(self, n_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Per-feature GRN transforms
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_features)
        ])

        # Selection weights GRN
        self.selection_grn = GatedResidualNetwork(
            n_features * hidden_dim, hidden_dim, n_features, dropout=dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        B, T, F = x.shape

        # Process each feature through its own GRN
        processed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]  # (B, T, 1)
            feat_i = feat_i.reshape(B * T, 1)
            out_i = self.feature_grns[i](feat_i)  # (B*T, hidden_dim)
            processed.append(out_i.reshape(B, T, self.hidden_dim))

        # Stack: (B, T, n_features, hidden_dim)
        stacked = torch.stack(processed, dim=2)

        # Flatten for selection weights
        flat = stacked.reshape(B, T, -1)  # (B, T, n_features * hidden_dim)
        flat = flat.reshape(B * T, -1)

        # Selection weights
        weights = self.selection_grn(flat)  # (B*T, n_features)
        weights = F.softmax(weights, dim=-1)  # (B*T, n_features)
        weights = weights.reshape(B, T, self.n_features, 1)

        # Weighted sum
        selected = (stacked * weights).sum(dim=2)  # (B, T, hidden_dim)
        return selected


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable weights.

    Uses additive attention following the TFT paper, producing
    per-head attention weights that can be averaged for interpretation.
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        H, D = self.n_heads, self.d_k

        # Project and reshape
        Q = self.W_q(q).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(k).reshape(B, T, H, D).transpose(1, 2)
        V = self.W_v(v).reshape(B, T, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, hidden_dim)
        return self.W_o(out)


# ── TFT Model ──────────────────────────────────────────────────────────────

class TFTModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer for d_iv prediction.

    Uses only observed (real-valued) inputs — no static covariates or
    known future inputs, as the task is next-day IV change prediction
    from a historical sequence window.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_dim : int
        Hidden dimension throughout the model.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of self-attention + GRN blocks.
    dropout : float
        Dropout rate.
    seed : int
        Random seed.
    """

    def __init__(self, n_features, hidden_dim=64, n_heads=4,
                 num_layers=1, dropout=0.1, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        # Variable selection
        self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout=dropout)

        # Temporal processing via LSTM (locality-enhancing layer in TFT)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Self-attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_grns = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout))
            self.attn_grns.append(
                GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout))
            self.attn_norms.append(nn.LayerNorm(hidden_dim))

        # Output
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, lookback, n_features)

        # Variable selection
        selected = self.vsn(x)  # (B, T, hidden_dim)

        # LSTM temporal processing
        lstm_out, _ = self.lstm(selected)
        gated = self.lstm_gate(lstm_out)
        temporal = self.lstm_norm(selected + gated)

        # Self-attention blocks
        h = temporal
        for attn, grn, norm in zip(self.attn_layers, self.attn_grns, self.attn_norms):
            attn_out = attn(h, h, h)
            h = norm(h + grn(attn_out))

        # Use last timestep for prediction
        last = h[:, -1, :]  # (B, hidden_dim)
        out = self.output_grn(last)
        return self.fc_out(out)  # (B, 1)
