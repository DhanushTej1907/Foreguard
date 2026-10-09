"""Learned forecasting baselines for the GCN-LSTM.

LSTMOnly: the GNN's temporal path without the GCN — each link's own edge features go
through a shared projection, LSTM and head. Same interface as CongestionPredictor.

PerLinkLogReg: one standardized logistic regression per link on that link's flattened
window of edge features (window_size x edge_feature_dim).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import MODEL


class LSTMOnly(nn.Module):
    def __init__(self, edge_feat_dim=None, lstm_hidden_dim=None, num_lstm_layers=None, dropout=None):
        super().__init__()
        edge_feat_dim = edge_feat_dim or MODEL["edge_feature_dim"]
        hidden = lstm_hidden_dim or MODEL["lstm_hidden_dim"]
        layers = num_lstm_layers or MODEL["num_lstm_layers"]
        dropout = dropout if dropout is not None else MODEL["dropout"]
        self.edge_proj = nn.Linear(edge_feat_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, node_features, edge_features, edge_index=None):
        """node_features and edge_index are ignored. edge_features: [B, W, E, F] -> [B, E]."""
        B, W, E, _ = edge_features.shape
        x = self.edge_proj(edge_features.permute(0, 2, 1, 3)).reshape(B * E, W, -1)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :])).reshape(B, E)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PerLinkLogReg:
    def fit(self, edge_features, labels):
        """edge_features: [N, W, E, F]; labels: [N, E]."""
        x = self._flatten(edge_features)
        self.models = []
        for e in range(x.shape[1]):
            y = labels[:, e]
            if np.unique(y).size < 2:
                self.models.append(float(y.mean()))   # single-class link: constant prior
            else:
                self.models.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                                   .fit(x[:, e], y))
        return self

    def predict_proba(self, edge_features):
        x = self._flatten(edge_features)
        out = np.empty(x.shape[:2])
        for e, m in enumerate(self.models):
            out[:, e] = m if isinstance(m, float) else m.predict_proba(x[:, e])[:, 1]
        return out

    @staticmethod
    def _flatten(edge_features):
        n, w, e, f = edge_features.shape
        return np.asarray(edge_features).transpose(0, 2, 1, 3).reshape(n, e, w * f)
