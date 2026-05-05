"""
dl_architectures.py — all PyTorch model classes for RUL prediction.

Extracted from individual experiment notebooks so they can be imported
without re-defining the architecture in every notebook.

Point-prediction models (output scalar RUL):
    GRUModel, LSTMModel, RNNModel, MLP, TransformerModel

Quantile models (output 3-vector [Q10, Q50, Q90]):
    QuantileGRU, QuantileLSTM, QuantileRNN, QuantileMLP, QuantileTransformer

Utility block:
    StableLSTMBlock  — LSTM + LayerNorm + residual (fixes Q_LSTM instability)

Usage
-----
    from src.models.dl_architectures import GRUModel, QuantileTransformer
    model = GRUModel(n_features=14)
    q_model = QuantileTransformer(n_features=14)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import DL_CONFIG

WINDOW_SIZE = DL_CONFIG["window_size"]


# ══════════════════════════════════════════════════════════════════════════════
# STABLE LSTM BLOCK (shared utility)
# ══════════════════════════════════════════════════════════════════════════════


class StableLSTMBlock(nn.Module):
    """
    LSTM + LayerNorm + residual projection.

    Why Q_LSTM failed (RMSE=40, bias=-27, coverage=21%):
        Pinball loss has steeper gradients for under-predictions (early RUL).
        Without layer normalization, hidden states can scale inconsistently
        across the 6 operating conditions of FD004 → the LSTM saturates
        into a low-RUL attractor → systematic under-prediction → narrow,
        over-confident low intervals → 21% coverage.

    Fix: LayerNorm after each LSTM output normalises hidden states across
    the feature dimension, preventing the low-RUL attractor.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = (
            nn.Linear(input_size, hidden_size, bias=False)
            if input_size != hidden_size else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        out     = self.norm(out)
        last    = out[:, -1, :]
        res     = self.proj(x[:, -1, :])
        return last + res


# ══════════════════════════════════════════════════════════════════════════════
# POINT-PREDICTION MODELS
# ══════════════════════════════════════════════════════════════════════════════


class GRUModel(nn.Module):
    """
    GRU for RUL point regression.

    Architecture: GRU (2 layers, hidden=64) → last hidden state → Linear(1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    """
    LSTM for RUL point regression.

    Architecture: LSTM (2 layers, hidden=64) → LayerNorm → last hidden state → Linear(1)

    LayerNorm added after LSTM output: without it, hidden states scale inconsistently
    across FD004's 6 operating conditions under NASA asymmetric loss, causing the model
    to saturate into a low-RUL attractor (bias ≈ −21, RMSE ≈ 32). LayerNorm prevents
    this by normalising hidden states across the feature dimension at each step.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.norm(out[:, -1, :])).squeeze(-1)


class RNNModel(nn.Module):
    """
    Vanilla RNN for RUL point regression.

    Architecture: RNN (2 layers, hidden=64) → last hidden state → Linear(1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.norm(out)
        return self.fc(out[:, -1, :]).squeeze(-1)


class MLP(nn.Module):
    """
    MLP for RUL point regression (no recurrence).

    WHY include: strongest non-temporal baseline.
    If MLP ≈ LSTM, the temporal structure isn't helping.

    Architecture: flatten window → FC(128) → ReLU → Dropout → FC(128) → ReLU → Dropout → FC(1)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = WINDOW_SIZE,
        hidden_size: int = 128,
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * window_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1)).squeeze(-1)


class TransformerModel(nn.Module):
    """
    Encoder-only Transformer for RUL point regression.

    Architecture:
        Linear(n_features → d_model) + learnable positional encoding
        → TransformerEncoder (n_heads=4, dim_ff=d_model×4, n_layers=2)
        → mean pool
        → Linear(d_model → 1)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = DL_CONFIG["d_model"],
        n_heads: int = DL_CONFIG["n_heads"],
        n_layers: int = DL_CONFIG["n_layers"],
        dropout: float = 0.1,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = nn.Embedding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * DL_CONFIG["dim_feedforward_mult"],
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        pos = torch.arange(W, device=x.device).unsqueeze(0)
        x   = self.input_proj(x) + self.pos_enc(pos)
        x   = self.transformer(x)
        x   = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# QUANTILE MODELS (output Q10, Q50, Q90)
# ══════════════════════════════════════════════════════════════════════════════


class QuantileGRU(nn.Module):
    """GRU with 3-output quantile head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        n_quantiles: int = 3,
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class QuantileLSTM(nn.Module):
    """
    LSTM with 3-output quantile head + LayerNorm.

    WHY LayerNorm: without it, LSTM hidden states scale inconsistently across
    FD004's 6 operating conditions under asymmetric pinball loss, causing
    systematic under-prediction and 21% coverage (Q_LSTM failure).
    LayerNorm normalises hidden states per-step, preventing saturation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        n_quantiles: int = 3,
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc   = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.norm(out[:, -1, :]))


class QuantileRNN(nn.Module):
    """
    RNN with 3-output quantile head + LayerNorm.

    LayerNorm added after RNN output: without it, Q10 collapses to 0 for all
    248 test engines under pinball loss on FD004's 6 operating conditions.
    The RNN hidden state scales inconsistently across conditions, causing the
    Q10 head to learn the trivial lower-bound solution (predict 0 always).
    LayerNorm normalises hidden states per-step, preventing this collapse.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = DL_CONFIG["hidden_size"],
        n_layers: int = DL_CONFIG["n_layers"],
        n_quantiles: int = 3,
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.rnn = nn.RNN(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(self.norm(out[:, -1, :]))


class QuantileMLP(nn.Module):
    """MLP with 3-output quantile head."""

    def __init__(
        self,
        n_features: int,
        window_size: int = WINDOW_SIZE,
        hidden_size: int = 128,
        n_quantiles: int = 3,
        dropout: float = DL_CONFIG["dropout"],
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * window_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class QuantileTransformer(nn.Module):
    """Encoder-only Transformer with 3-output quantile head."""

    def __init__(
        self,
        n_features: int,
        d_model: int = DL_CONFIG["d_model"],
        n_heads: int = DL_CONFIG["n_heads"],
        n_layers: int = DL_CONFIG["n_layers"],
        n_quantiles: int = 3,
        dropout: float = 0.1,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.pos_enc     = nn.Embedding(window_size, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * DL_CONFIG["dim_feedforward_mult"],
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc          = nn.Linear(d_model, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        pos = torch.arange(W, device=x.device).unsqueeze(0)
        x   = self.input_proj(x) + self.pos_enc(pos)
        x   = self.transformer(x).mean(dim=1)
        return self.fc(x)


# ── Model registry ──────────────────────────────────────────────────────────────
# Map name → class for programmatic instantiation in pipeline.
POINT_MODELS: dict[str, type] = {
    "GRU":         GRUModel,
    "LSTM":        LSTMModel,
    "RNN":         RNNModel,
    "MLP":         MLP,
    "Transformer": TransformerModel,
}

QUANTILE_MODELS: dict[str, type] = {
    "Q_GRU":         QuantileGRU,
    "Q_LSTM":        QuantileLSTM,
    "Q_RNN":         QuantileRNN,
    "Q_MLP":         QuantileMLP,
    "Q_Transformer": QuantileTransformer,
}

ALL_MODELS = {**POINT_MODELS, **QUANTILE_MODELS}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    """
    Instantiate any registered model by name.

    Usage:
        model = build_model("GRU", n_features=14)
        q_model = build_model("Q_Transformer", n_features=14)
    """
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ALL_MODELS)}")
    return ALL_MODELS[name](n_features=n_features, **kwargs)
