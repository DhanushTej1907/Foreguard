"""
GCN-LSTM Congestion Prediction Model
=====================================
Architecture:
  1. For each timestep in the window:
     a. Node features → 2× GCNConv layers → node embeddings
     b. For each edge: concatenate [src_embed, dst_embed, edge_features] → edge embedding
  2. Stack edge embeddings across time → LSTM → temporal edge embedding
  3. MLP head → per-edge congestion probability (sigmoid)

This captures BOTH:
  - Spatial dependencies (which neighboring links affect each other) via GCN
  - Temporal dynamics (how congestion builds over time) via LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Try to import PyG; fall back to manual GCN if not available
try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from config import MODEL, CHECKPOINT_SCHEMA_VERSION


def check_checkpoint_schema(checkpoint: dict):
    """Reject checkpoints trained on an older simulator/feature/label schema or another label mode."""
    found = checkpoint.get("schema_version")
    if found != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Legacy checkpoint (schema {found}, need {CHECKPOINT_SCHEMA_VERSION}): "
                         "regenerate data and retrain")
    if checkpoint.get("label_mode") != MODEL["label_mode"]:
        raise ValueError(f"Checkpoint label_mode {checkpoint.get('label_mode')!r} != configured "
                         f"{MODEL['label_mode']!r}: retrain")


class ManualGCNConv(nn.Module):
    """
    Manual GCN convolution layer (fallback when PyTorch Geometric is not installed).
    Implements: H' = σ(D^{-1/2} A D^{-1/2} H W)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        """
        x: [N, in_channels]
        edge_index: [2, E] (COO format)
        """
        N = x.size(0)

        # Transform features
        x = x @ self.weight  # [N, out_channels]

        # Build sparse adjacency with self-loops
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops
        self_loops = torch.arange(N, device=x.device)
        src = torch.cat([src, self_loops])
        dst = torch.cat([dst, self_loops])

        # Compute degree for normalization
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Message passing: aggregate neighbor features
        out = torch.zeros_like(x)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # [E]

        # Scatter add
        messages = x[src] * norm.unsqueeze(-1)  # [E, out_channels]
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        return out + self.bias


def load_compatible_state_dict(model: nn.Module, state_dict: dict, strict: bool = True):
    """Load checkpoints across the optional PyG/manual GCN backends.

    Older checkpoints may contain PyG keys (``*.lin.weight``), while a
    lightweight environment can use the bundled ManualGCNConv implementation.
    ``torch.compile`` can also add an ``_orig_mod.`` prefix. Normalize those
    representations so the same trusted checkpoint remains portable.
    """
    target_model = getattr(model, "_orig_mod", model)
    target_keys = set(target_model.state_dict().keys())
    normalized = OrderedDict()
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.")
        if key not in target_keys:
            if key.endswith(".lin.weight"):
                manual_key = key[:-len(".lin.weight")] + ".weight"
                if manual_key in target_keys:
                    key = manual_key
                    value = value.transpose(0, 1).contiguous()
            elif key.endswith(".weight") and ".layers." in key:
                pyg_key = key[:-len(".weight")] + ".lin.weight"
                if pyg_key in target_keys:
                    key = pyg_key
                    value = value.transpose(0, 1).contiguous()
        normalized[key] = value
    return target_model.load_state_dict(normalized, strict=strict)


class GCNEncoder(nn.Module):
    """
    Spatial encoder: processes one timestep's graph through GCN layers.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        ConvLayer = GCNConv if HAS_PYG else ManualGCNConv

        # First layer
        self.layers.append(ConvLayer(node_feat_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ConvLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        x: [N, node_feat_dim]
        edge_index: [2, E]
        Returns: [N, hidden_dim]
        """
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CongestionPredictor(nn.Module):
    """
    Full GCN-LSTM model for per-link congestion prediction.

    Input:  node features [B, W, N, F_node], edge features [B, W, E, F_edge]
    Output: congestion probability per edge [B, E]
    """

    def __init__(self, node_feat_dim: int = None, edge_feat_dim: int = None,
                 hidden_dim: int = None, lstm_hidden_dim: int = None,
                 num_gcn_layers: int = None, num_lstm_layers: int = None,
                 dropout: float = None, num_edges: int = None,
                 edge_pairs: list = None):
        super().__init__()

        # Defaults from config
        node_feat_dim = node_feat_dim or MODEL["node_feature_dim"]
        edge_feat_dim = edge_feat_dim or MODEL["edge_feature_dim"]
        hidden_dim = hidden_dim or MODEL["hidden_dim"]
        lstm_hidden_dim = lstm_hidden_dim or MODEL["lstm_hidden_dim"]
        num_gcn_layers = num_gcn_layers or MODEL["num_gcn_layers"]
        num_lstm_layers = num_lstm_layers or MODEL["num_lstm_layers"]
        dropout = dropout if dropout is not None else MODEL["dropout"]

        # Store edge pair info for converting node embeddings → edge embeddings
        self.num_edges = num_edges
        self.edge_pairs = edge_pairs  # List of (src_idx, dst_idx) for undirected edges

        # Spatial encoder
        self.gcn = GCNEncoder(node_feat_dim, hidden_dim, num_gcn_layers, dropout)

        # Edge embedding: concat(src_node_embed, dst_node_embed, edge_features)
        edge_embed_dim = hidden_dim * 2 + edge_feat_dim

        # Edge feature projector
        self.edge_proj = nn.Linear(edge_embed_dim, lstm_hidden_dim)

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Prediction head (per edge)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, 1),
        )

    def forward(self, node_features, edge_features, edge_index):
        """
        Parameters
        ----------
        node_features : [B, W, N, F_node]
        edge_features : [B, W, E_undirected, F_edge]
        edge_index    : [2, E_directed]  (COO, bidirectional)

        Returns
        -------
        pred : [B, E_undirected]  congestion probability per edge
        """
        B, W, N, _ = node_features.shape
        E = edge_features.shape[2]  # Number of undirected edges

        # Fully vectorize GCN by flattening all graphs into one large disjoint graph
        node_features_flat = node_features.view(B * W * N, -1)
        
        # Shift edge indices to create a single large disjoint graph
        # edge_index is [2, E_directed]
        E_dir = edge_index.shape[1]
        offsets = torch.arange(B * W, device=node_features.device) * N
        edge_index_batched = edge_index.unsqueeze(0).repeat(B * W, 1, 1) # [B*W, 2, E_dir]
        edge_index_batched += offsets.view(B * W, 1, 1)
        edge_index_batched = edge_index_batched.transpose(0, 1).reshape(2, B * W * E_dir) # [2, B*W*E_dir]

        # Single GCN pass for the entire batch and all timesteps!
        node_embed_flat = self.gcn(node_features_flat, edge_index_batched)
        
        node_embed = node_embed_flat.view(B, W, N, -1) # [B, W, N, hidden]

        # Vectorize edge embeddings across B, W, and E simultaneously
        if self.edge_pairs is not None:
            src_indices = [p[0] for p in self.edge_pairs]
            dst_indices = [p[1] for p in self.edge_pairs]
        else:
            src_indices = edge_index[0, :E*2:2]
            dst_indices = edge_index[1, :E*2:2]

        src_embeds = node_embed[:, :, src_indices, :]  # [B, W, E, hidden]
        dst_embeds = node_embed[:, :, dst_indices, :]  # [B, W, E, hidden]
        
        # Concat: [src | dst | edge_features]
        # shape: [B, W, E, 2*hidden + F_edge]
        edge_temporal = torch.cat([src_embeds, dst_embeds, edge_features], dim=-1)
        
        # Permute to [B, E, W, dim] for LSTM
        edge_temporal = edge_temporal.permute(0, 2, 1, 3)

        # Already permuted to [B, E, W, embed_dim]

        # Project edge embeddings
        edge_temporal = self.edge_proj(edge_temporal)  # [B, E, W, lstm_hidden]

        # Process each edge's time series through LSTM
        B, E, W, D = edge_temporal.shape
        edge_temporal = edge_temporal.reshape(B * E, W, D)  # [B*E, W, lstm_hidden]

        lstm_out, _ = self.lstm(edge_temporal)  # [B*E, W, lstm_hidden]
        last_hidden = lstm_out[:, -1, :]        # [B*E, lstm_hidden]

        # Prediction head
        pred = self.head(last_hidden)            # [B*E, 1]
        pred = pred.reshape(B, E)                # [B, E]
        pred = torch.sigmoid(pred)               # [B, E] ∈ [0, 1]

        return pred

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
