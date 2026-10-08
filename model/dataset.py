"""
Dataset Builder
===============
Converts simulation telemetry into PyTorch Geometric Data objects
for training the GNN congestion predictor.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from config import MODEL, DATA_DIR, ensure_dirs


class CongestionDataset(Dataset):
    """
    PyTorch Dataset for congestion prediction.

    Each sample contains:
      - node_features: [window_size, num_nodes, node_feat_dim]
      - edge_features: [window_size, num_edges, edge_feat_dim]
      - edge_index:    [2, num_directed_edges]  (COO format)
      - labels:        [num_edges]  (binary congestion at t + horizon)
    """

    def __init__(self, samples: list, edge_index: np.ndarray):
        """
        Parameters
        ----------
        samples : list of dicts
            Each with 'node_features', 'edge_features', 'labels' numpy arrays.
        edge_index : np.ndarray
            Shape [2, num_directed_edges], COO format for PyG.
        """
        self.samples = samples
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "node_features": torch.tensor(s["node_features"], dtype=torch.float32),
            "edge_features": torch.tensor(s["edge_features"], dtype=torch.float32),
            "edge_index": self.edge_index,
            "labels": torch.tensor(s["labels"], dtype=torch.float32),
        }


def build_datasets(all_samples: list, edge_index: np.ndarray,
                   train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Split samples into train/val/test datasets.

    Splits whole scenarios so overlapping windows never cross partitions.
    """
    from model.splits import split_scenarios
    train, val, test = split_scenarios(all_samples, train_frac, val_frac)
    train_ds = CongestionDataset(train, edge_index)
    val_ds = CongestionDataset(val, edge_index)
    test_ds = CongestionDataset(test, edge_index)

    print(f"  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds


def save_datasets(train_ds, val_ds, test_ds, filename="datasets.pkl"):
    """Save datasets to disk."""
    ensure_dirs()
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump({"train": train_ds, "val": val_ds, "test": test_ds}, f)
    print(f"  Saved datasets to {path}")


def load_datasets(filename="datasets.pkl"):
    """Load datasets from disk."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["train"], data["val"], data["test"]


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Stacks temporal graph samples into a batch.
    """
    node_features = torch.stack([b["node_features"] for b in batch])
    edge_features = torch.stack([b["edge_features"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    edge_index = batch[0]["edge_index"]  # Same for all samples

    return {
        "node_features": node_features,     # [B, W, N, F_node]
        "edge_features": edge_features,     # [B, W, E, F_edge]
        "edge_index": edge_index,           # [2, num_directed_edges]
        "labels": labels,                   # [B, E]
    }
