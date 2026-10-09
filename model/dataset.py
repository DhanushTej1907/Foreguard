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
from config import MODEL, DATA_OUT, CHECKPOINT_SCHEMA_VERSION, check_output_path, ensure_dirs


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
            "last_labels": torch.tensor(s["last_labels"], dtype=torch.float32),
            "last_scores": torch.tensor(s["last_scores"], dtype=torch.float32),
        }

    def stacked(self, key):
        """All samples' `key` arrays stacked into one numpy array."""
        return np.stack([s[key] for s in self.samples])


def build_datasets(all_samples: list, edge_index: np.ndarray,
                   train_frac: float = 0.7, val_frac: float = 0.15,
                   holdout_family: str = None, split: str = "random"):
    """
    Split samples into train/val/test datasets.

    Splits whole scenarios so overlapping windows never cross partitions. With
    `holdout_family`, the test set is every scenario of that traffic family; with
    split="stratified", every family appears in every partition.
    """
    from model.splits import split_scenarios, holdout_split, stratified_split
    if holdout_family is not None:
        train, val, test = holdout_split(all_samples, holdout_family)
    elif split == "stratified":
        train, val, test = stratified_split(all_samples)
    elif split == "random":
        train, val, test = split_scenarios(all_samples, train_frac, val_frac)
    else:
        raise ValueError(f"unknown split {split!r}")
    train_ds = CongestionDataset(train, edge_index)
    val_ds = CongestionDataset(val, edge_index)
    test_ds = CongestionDataset(test, edge_index)

    print(f"  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds


def save_datasets(train_ds, val_ds, test_ds, path=None):
    """Save datasets with the schema and label mode they were generated under."""
    ensure_dirs()
    path = check_output_path(path or DATA_OUT)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"schema_version": CHECKPOINT_SCHEMA_VERSION, "label_mode": MODEL["label_mode"],
                     "train": train_ds, "val": val_ds, "test": test_ds}, f)
    print(f"  Saved datasets to {path}")


def check_dataset_schema(data: dict):
    """Reject datasets generated under another schema or label mode."""
    found = (data.get("schema_version"), data.get("label_mode"))
    expected = (CHECKPOINT_SCHEMA_VERSION, MODEL["label_mode"])
    if found != expected:
        raise ValueError(f"Dataset has schema/label_mode {found}, need {expected}: regenerate data")


def load_datasets(path=None):
    """Load datasets from disk, rejecting mismatched schema or label mode."""
    with open(path or DATA_OUT, "rb") as f:
        data = pickle.load(f)
    check_dataset_schema(data)
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
        "last_labels": torch.stack([b["last_labels"] for b in batch]),  # [B, E]
        "last_scores": torch.stack([b["last_scores"] for b in batch]),  # [B, E]
    }
