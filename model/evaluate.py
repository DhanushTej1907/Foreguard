"""
Evaluation & Visualization
===========================
Evaluates the trained GCN-LSTM model and generates publication-quality plots:
  - Confusion matrix
  - ROC-AUC curve
  - Precision-Recall curve
  - Training history curves
  - Strategy comparison (No Rerouting vs Reactive vs Proactive)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PLOTS_DIR, ensure_dirs
from model.dataset import collate_fn


# ── Plot Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0e1a",
    "axes.facecolor": "#0f1729",
    "axes.edgecolor": "#1e293b",
    "axes.labelcolor": "#cbd5e1",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#1e293b",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})

COLORS = {
    "primary": "#00d2ff",
    "secondary": "#e94560",
    "accent": "#f6ad55",
    "success": "#4fd1c5",
    "warning": "#f093fb",
    "indigo": "#7b68ee",
    "green": "#48bb78",
    "bg": "#0a0e1a",
    "card": "#0f1729",
}


def evaluate_model(model, test_ds, device="cpu"):
    """
    Evaluate model on test set and return predictions + metrics.
    """
    model.eval()
    model = model.to(device)

    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            node_feat = batch["node_features"].to(device)
            edge_feat = batch["edge_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            labels = batch["labels"].to(device)

            probs = model(node_feat, edge_feat, edge_index)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend((probs > 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print classification report
    print("\n" + "="*50)
    print("  Classification Report")
    print("="*50)
    print(classification_report(all_labels, all_preds, labels=[0, 1],
                                 target_names=["Normal", "Congested"],
                                 zero_division=0))

    return all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, save_path=None):
    """Plot confusion matrix with premium styling."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("neural", [
        "#0f1729", "#1a365d", "#2b6cb0", "#00d2ff"
    ])

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold",
                  color=COLORS["primary"], pad=15)

    classes = ["Normal", "Congested"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#94a3b8",
                    fontsize=20, fontweight="bold")

    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors="#64748b")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
    plt.close()
    return fig


def plot_roc_curve(labels, probs, save_path=None):
    """Plot ROC-AUC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Filled area
    ax.fill_between(fpr, tpr, alpha=0.12, color=COLORS["primary"])

    # ROC curve
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5, alpha=0.9,
            label=f"GCN-LSTM (AUC = {roc_auc:.3f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], color=COLORS["secondary"], lw=1.5,
            linestyle="--", alpha=0.5, label="Random (AUC = 0.500)")

    # AUC annotation
    ax.annotate(f"AUC = {roc_auc:.3f}",
                xy=(0.5, 0.3), fontsize=24, fontweight="bold",
                color=COLORS["primary"], alpha=0.15,
                ha="center", va="center")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold",
                  color=COLORS["primary"], pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.3,
              facecolor=COLORS["card"])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
    plt.close()
    return fig


def plot_precision_recall(labels, probs, save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.fill_between(recall, precision, alpha=0.12, color=COLORS["accent"])
    ax.plot(recall, precision, color=COLORS["accent"], lw=2.5, alpha=0.9,
            label=f"GCN-LSTM (AUC = {pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold",
                  color=COLORS["accent"], pad=15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.3,
              facecolor=COLORS["card"])
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
    plt.close()
    return fig


def plot_training_history(history, save_path=None):
    """Plot training curves (loss, F1, precision/recall) side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──
    axes[0].plot(epochs, history["train_loss"], color=COLORS["primary"],
                 lw=2, label="Train", alpha=0.9)
    axes[0].plot(epochs, history["val_loss"], color=COLORS["secondary"],
                 lw=2, label="Validation", alpha=0.9)
    axes[0].fill_between(epochs, history["train_loss"], alpha=0.05,
                          color=COLORS["primary"])
    axes[0].fill_between(epochs, history["val_loss"], alpha=0.05,
                          color=COLORS["secondary"])
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=13,
                       fontweight="bold", color=COLORS["primary"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(framealpha=0.3, facecolor=COLORS["card"])
    axes[0].grid(True, alpha=0.15)

    # ── F1 Score ──
    axes[1].plot(epochs, history["train_f1"], color=COLORS["primary"],
                 lw=2, label="Train", alpha=0.9)
    axes[1].plot(epochs, history["val_f1"], color=COLORS["secondary"],
                 lw=2, label="Validation", alpha=0.9)
    axes[1].fill_between(epochs, history["train_f1"], alpha=0.05,
                          color=COLORS["primary"])
    axes[1].fill_between(epochs, history["val_f1"], alpha=0.05,
                          color=COLORS["secondary"])
    axes[1].set_title("F1 Score", fontsize=13, fontweight="bold",
                       color=COLORS["accent"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend(framealpha=0.3, facecolor=COLORS["card"])
    axes[1].grid(True, alpha=0.15)

    # ── Precision & Recall ──
    axes[2].plot(epochs, history["val_precision"], color=COLORS["success"],
                 lw=2, label="Precision", alpha=0.9)
    axes[2].plot(epochs, history["val_recall"], color=COLORS["warning"],
                 lw=2, label="Recall", alpha=0.9)
    axes[2].fill_between(epochs, history["val_precision"], alpha=0.05,
                          color=COLORS["success"])
    axes[2].fill_between(epochs, history["val_recall"], alpha=0.05,
                          color=COLORS["warning"])
    axes[2].set_title("Precision & Recall", fontsize=13, fontweight="bold",
                       color=COLORS["success"])
    axes[2].set_xlabel("Epoch")
    axes[2].legend(framealpha=0.3, facecolor=COLORS["card"])
    axes[2].grid(True, alpha=0.15)

    plt.suptitle("Training History — GCN-LSTM Congestion Predictor",
                  fontsize=16, fontweight="bold", color=COLORS["primary"],
                  y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
    plt.close()
    return fig


def plot_comparison(comparison_results, save_path=None):
    """
    Plot side-by-side comparison of routing strategies with
    premium grouped bar charts.

    comparison_results: dict with keys "No Rerouting", "Reactive", "Proactive (GNN)"
    Each contains: avg_latency, max_latency, total_drops, avg_throughput
    """
    strategies = list(comparison_results.keys())
    metrics = ["avg_latency", "max_latency", "total_drops", "avg_throughput"]
    titles = [
        "Mean link queue-delay proxy",
        "Peak link queue-delay proxy",
        "Packet Drops (lower = better)",
        "Delivered packets per tick",
    ]
    units = ["μs", "μs", "packets", "pkt/tick"]

    colors = [COLORS["secondary"], COLORS["accent"], COLORS["primary"]]
    if len(strategies) > 3:
        colors.append(COLORS["indigo"])

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for i, (metric, title, unit) in enumerate(zip(metrics, titles, units)):
        values = [comparison_results[s].get(metric, 0) for s in strategies]

        bars = axes[i].bar(
            range(len(strategies)), values,
            color=colors[:len(strategies)],
            edgecolor="none",
            width=0.55,
            alpha=0.85,
            zorder=3,
        )

        # Value labels
        for bar, val in zip(bars, values):
            y = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, y + y * 0.02,
                f"{val:,.1f}" if val > 10 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=10,
                color=COLORS["primary"], fontweight="bold"
            )

        axes[i].set_title(title, fontsize=11, fontweight="bold",
                           color=COLORS["primary"], pad=12)
        axes[i].set_ylabel(f"{unit}", fontsize=10)
        axes[i].set_xticks(range(len(strategies)))
        axes[i].set_xticklabels(
            [s.replace(" (GNN)", "\n(GNN)") for s in strategies],
            fontsize=9
        )
        axes[i].grid(True, axis="y", alpha=0.12, zorder=0)
        axes[i].set_axisbelow(True)

        # Highlight best (lowest for latency/drops, highest for throughput)
        if metric in ("avg_latency", "max_latency", "total_drops"):
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor(COLORS["green"])
        bars[best_idx].set_linewidth(2)
        bars[best_idx].set_alpha(1.0)

    plt.suptitle("Routing Strategy Comparison — GNN Proactive vs Baselines",
                  fontsize=16, fontweight="bold", color=COLORS["primary"],
                  y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
    plt.close()
    return fig


def generate_all_plots(model, test_ds, history, comparison_results=None,
                       device="cpu"):
    """Generate all evaluation plots and save to PLOTS_DIR."""
    ensure_dirs()

    print("\n[*] Generating evaluation plots...")

    labels, preds, probs = evaluate_model(model, test_ds, device)

    # 1. Confusion Matrix
    plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )
    print("  [OK] Confusion matrix")

    # 2. ROC-AUC
    plot_roc_curve(
        labels, probs,
        save_path=os.path.join(PLOTS_DIR, "roc_curve.png")
    )
    print("  [OK] ROC curve")

    # 3. Precision-Recall
    plot_precision_recall(
        labels, probs,
        save_path=os.path.join(PLOTS_DIR, "precision_recall.png")
    )
    print("  [OK] Precision-Recall curve")

    # 4. Training history
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(PLOTS_DIR, "training_history.png")
        )
        print("  [OK] Training history")

    # 5. Comparison
    if comparison_results:
        plot_comparison(
            comparison_results,
            save_path=os.path.join(PLOTS_DIR, "comparison.png")
        )
        print("  [OK] Strategy comparison")

    print(f"\n  All plots saved to: {PLOTS_DIR}")
