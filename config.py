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
    "warmup_ticks": 20,              # Warmup ticks before recording telemetry
    "num_scenarios": 10,             # Number of traffic scenarios to generate
}

# ─── Traffic Patterns ─────────────────────────────────────────
TRAFFIC = {
    "baseline_utilization": 0.4,     # Average baseline link utilization
    "hotspot_intensity": 0.95,       # Peak utilization during hotspot events
    "hotspot_probability": 0.15,     # Probability of hotspot event per tick
    "elephant_flow_rate": 0.8,       # Elephant flow rate (fraction of link capacity)
    "mice_flow_rate": 0.15,          # Mice flow rate (fraction of link capacity)
    "elephant_fraction": 0.2,        # Fraction of flows that are elephants
    "incast_degree": 6,              # Number of sources in incast pattern
}

# ─── GNN Model ────────────────────────────────────────────────────
MODEL = {
    "node_feature_dim": 4,           # Queue occupancy, throughput, drop rate, buffer usage
    "edge_feature_dim": 4,           # Utilization, queue depth, drop rate, latency
    "hidden_dim": 64,                # GCN hidden dimension
    "lstm_hidden_dim": 32,           # LSTM hidden dimension
    "num_gcn_layers": 2,             # Number of GCN layers
    "num_lstm_layers": 1,            # Number of LSTM layers
    "dropout": 0.3,                  # Dropout rate
    "window_size": 10,               # Temporal window (number of past timesteps)
    "prediction_horizon": 5,         # Predict congestion Δt ticks ahead
    "congestion_threshold": 0.85,    # Utilization above this = congested
}

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
    "train_split": 0.7,              # Train fraction
    "val_split": 0.15,               # Validation fraction
    "test_split": 0.15,              # Test fraction
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
