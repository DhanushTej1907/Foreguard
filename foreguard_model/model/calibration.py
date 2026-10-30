"""Temperature scaling for sigmoid outputs, fitted on validation (NLL).

p_cal = sigmoid(logit(p) / T). Monotonic in p, so AP/AUC and ranking are unchanged;
only probability values (Brier, threshold location) move.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.special import expit

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def apply_temperature(probs, temperature):
    return expit(_logit(probs) / temperature)


def fit_temperature(probs, labels, bounds=(0.05, 20.0)):
    """Temperature minimizing binary NLL of the calibrated probabilities."""
    z = _logit(probs).ravel()
    y = np.asarray(labels, dtype=float).ravel()

    def nll(log_t):
        q = np.clip(expit(z / np.exp(log_t)), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=tuple(np.log(bounds)), method="bounded")
    return float(np.exp(res.x))


class TemperatureScaled(nn.Module):
    """Wraps a probability model; same forward signature, calibrated output."""

    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = float(temperature)

    def forward(self, node_features, edge_features, edge_index=None):
        p = self.model(node_features, edge_features, edge_index).clamp(EPS, 1 - EPS)
        return torch.sigmoid(torch.logit(p) / self.temperature)

    def count_parameters(self):
        return self.model.count_parameters()
