"""
Central configuration for the GNN Congestion Pre-emption System.
All hyperparameters and settings in one place.
"""

import os

# ─── Project Paths ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Default outputs for schema-3 artifacts. The legacy (schema-2) files are kept for
# reference and must never be overwritten.
CHECKPOINT_OUT = os.path.join(MODEL_DIR, "best_model_v3.pt")
RESULTS_OUT = os.path.join(PROJECT_ROOT, "research", "benchmark_results_v4.json")
EVALUATE_RESULTS_OUT = os.path.join(PROJECT_ROOT, "research", "evaluate_results_v4.json")
DATA_OUT = os.path.join(DATA_DIR, "datasets_v3.pkl")
PLOTS_OUT_DIR = os.path.join(PLOTS_DIR, "v3")
LEGACY_PATHS = [os.path.join(MODEL_DIR, "best_model.pt"),
                os.path.join(MODEL_DIR, "training_history.json"),
                os.path.join(PROJECT_ROOT, "research", "benchmark_results.json"),
                os.path.join(DATA_DIR, "datasets.pkl")]


def check_output_path(path):
    """Raise if `path` would overwrite a legacy schema-2 artifact; return it otherwise."""
    norm = lambda p: os.path.normcase(os.path.abspath(p))
    if norm(path) in {norm(p) for p in LEGACY_PATHS} or os.path.dirname(norm(path)) == norm(PLOTS_DIR):
        raise ValueError(f"refusing to overwrite legacy artifact {path}; choose a new filename")
    return path

# ─── Topology ─────────────────────────────────────────────────
TOPOLOGY = {
    "k": 4,                          # Fat-Tree parameter (k=4 -> 20 switches, 16 hosts)
    "link_capacity_gbps": 10.0,      # Link capacity in Gbps
    "link_capacity_packets_per_tick": 500,        # Packets per tick; independent of nominal Gbps metadata
    "propagation_delay_us": 0.5,     # Link propagation delay in microseconds
    "buffer_size_packets": 2000,     # Switch output buffer size in packets
}

# ─── Simulation ───────────────────────────────────────────────
SIMULATION = {
    "tick_duration_ms": 1.0,         # Time step in milliseconds
    "total_ticks": 500,              # Total simulation ticks for data generation
    "num_scenarios": 10,             # Number of traffic scenarios to generate
}

# ─── Traffic Patterns ─────────────────────────────────────────
TRAFFIC = {
    "baseline_utilization": 0.4,     # Average baseline link utilization
    "hotspot_intensity": 0.95,       # Peak utilization during hotspot events
    "elephant_flow_rate": 0.8,       # Elephant flow rate (fraction of link capacity)
    "mice_flow_rate": 0.15,          # Mice flow rate (fraction of link capacity)
    "elephant_fraction": 0.2,        # Fraction of flows that are elephants
    "incast_degree": 6,              # Number of sources in incast pattern
}

# ─── GNN Model ────────────────────────────────────────────────────
MODEL = {
    "node_feature_dim": 4,           # Queue occupancy, throughput, drop rate, buffer usage
    "edge_feature_dim": 4,           # Offered utilization, queue depth, drop rate, served utilization
    "hidden_dim": 64,                # GCN hidden dimension
    "lstm_hidden_dim": 32,           # LSTM hidden dimension
    "num_gcn_layers": 2,             # Number of GCN layers
    "num_lstm_layers": 1,            # Number of LSTM layers
    "dropout": 0.3,                  # Dropout rate
    "window_size": 10,               # Temporal window (number of past timesteps)
    "prediction_horizon": 5,         # Predict congestion Δt ticks ahead
    "congestion_threshold": 0.85,    # Utilization above this = congested
    # "offered":          packets_in / capacity > congestion_threshold
    # "served_or_queued": packets_out / capacity > congestion_threshold
    #                     OR queue > queue_label_fraction * buffer
    # Either direction of a physical link meeting the rule labels the link congested.
    # The label is defined as congestion_score > congestion_threshold (simulation/telemetry.py).
    "label_mode": "served_or_queued",
    "queue_label_fraction": 0.1,
}

# Bump whenever simulator, features or labels change; older checkpoints are rejected.
# Schema 3 = Phase 1 (full-duplex) + Phase 2 (served_or_queued default, served-utilization
# feature). No schema-3 artifact existed before Phase 2, so Phase 2 did not bump it.
CHECKPOINT_SCHEMA_VERSION = 3

# ─── Training ─────────────────────────────────────────────────────
TRAINING = {
    "epochs": 100,                   # Maximum training epochs
    "batch_size": 32,                # Batch size
    "learning_rate": 1e-3,           # Initial learning rate
    "weight_decay": 1e-4,            # L2 regularization
    "lr_patience": 10,               # Patience for LR scheduler
    "lr_factor": 0.5,                # LR reduction factor
    "early_stop_patience": 20,       # Early stopping patience
    "pos_weight": 3.0,               # Positive class weight (congestion is rarer)
}

# ─── Rerouting ────────────────────────────────────────────────────
REROUTING = {
    "congestion_alert_threshold": 0.7,   # GNN prediction threshold to trigger reroute
    "reactive_threshold": 0.9,           # Reactive baseline: reroute after this utilization
    "k_shortest_paths": 3,               # Number of alternative paths to consider
    "reroute_cooldown_ticks": 5,         # Minimum ticks between reroutes for same flow
}

# ─── Dashboard ────────────────────────────────────────────────────
DASHBOARD = {
    "host": "127.0.0.1",
    "port": 8000,
    "ws_update_interval_ms": 100,    # WebSocket push interval
}


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MODEL_DIR, DATA_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)
