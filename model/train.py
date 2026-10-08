"""
Training Loop
=============
Trains the GCN-LSTM congestion predictor with:
  - Weighted Binary Cross-Entropy loss (handles class imbalance)
  - Adam optimizer with ReduceLROnPlateau scheduler
  - Early stopping on validation F1-score
  - Checkpoint saving of best model
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from config import TRAINING, MODEL, MODEL_DIR, CHECKPOINT_SCHEMA_VERSION, ensure_dirs
from model.dataset import collate_fn
from model.gnn_model import load_compatible_state_dict


def train_model(model, train_ds, val_ds, device="cpu", epochs=None,
                verbose=True):
    """
    Train the congestion prediction model.

    Parameters
    ----------
    model : CongestionPredictor
    train_ds : CongestionDataset
    val_ds : CongestionDataset
    device : str
    epochs : int

    Returns
    -------
    dict : training history
    """
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    if not len(train_ds) or not len(val_ds):
        raise ValueError("training and validation datasets must be nonempty")
    ensure_dirs()
    if epochs is None:
        epochs = TRAINING["epochs"]

    model = model.to(device)

    # Keep checkpoints portable and deterministic by default.
    # Set NEURALFABRIC_TORCH_COMPILE=1 only for local performance experiments.
    if os.environ.get("NEURALFABRIC_TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = torch.compile(model)
            except Exception:
                pass

    # Loss: weighted BCE for class imbalance
    pos_weight = torch.tensor([TRAINING["pos_weight"]], device=device)
    criterion = nn.BCELoss(reduction="mean")

    # We'll apply pos_weight manually since BCELoss doesn't have pos_weight
    # Use BCEWithLogitsLoss instead (more numerically stable)
    # But our model outputs sigmoid, so we use BCE with manual weighting

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=TRAINING["lr_factor"],
        patience=TRAINING["lr_patience"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=TRAINING["batch_size"],
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAINING["batch_size"],
        shuffle=False, collate_fn=collate_fn,
    )

    # Training state
    best_f1 = -1.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_acc": [], "val_acc": [],
        "val_precision": [], "val_recall": [],
        "lr": [],
    }

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training GCN-LSTM Congestion Predictor")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Device: {device}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────────
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for batch in train_loader:
            node_feat = batch["node_features"].to(device)
            edge_feat = batch["edge_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            pred = model(node_feat, edge_feat, edge_index)

            # Weighted BCE loss
            weight = torch.where(labels == 1,
                                 pos_weight.expand_as(labels),
                                 torch.ones_like(labels))
            loss = F.binary_cross_entropy(pred, labels, weight=weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend((pred > 0.5).cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                node_feat = batch["node_features"].to(device)
                edge_feat = batch["edge_features"].to(device)
                edge_index = batch["edge_index"].to(device)
                labels = batch["labels"].to(device)

                pred = model(node_feat, edge_feat, edge_index)

                weight = torch.where(labels == 1,
                                     pos_weight.expand_as(labels),
                                     torch.ones_like(labels))
                loss = F.binary_cross_entropy(pred, labels, weight=weight)

                val_losses.append(loss.item())
                val_preds.extend((pred > 0.5).cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        # ── Metrics ───────────────────────────────────────────────
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["lr"].append(current_lr)

        scheduler.step(val_f1)

        # ── Checkpoint ────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "label_mode": MODEL["label_mode"],
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": float(val_f1),
                "val_loss": float(val_loss),
            }, checkpoint_path)
        else:
            patience_counter += 1

        elapsed = time.time() - t0

        if verbose and (epoch % 5 == 0 or epoch <= 3 or epoch == epochs):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"F1: {train_f1:.3f}/{val_f1:.3f} | "
                  f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
                  f"P/R: {val_prec:.3f}/{val_rec:.3f} | "
                  f"LR: {current_lr:.6f} | "
                  f"{elapsed:.1f}s")

        # ── Early Stopping ────────────────────────────────────────
        if patience_counter >= TRAINING["early_stop_patience"]:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} (best F1: {best_f1:.4f})")
            break

    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_compatible_state_dict(model, checkpoint["model_state_dict"])

    if verbose:
        print(f"\n  Best model: epoch {checkpoint['epoch']}, "
              f"val_f1={checkpoint['val_f1']:.4f}")
        print(f"  Saved to: {checkpoint_path}")

    return history

