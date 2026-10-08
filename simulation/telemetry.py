"""
INT-style idealized telemetry
=============================
Per-link and per-switch counters (queue depth, utilization, drops, latency) are
read directly from the simulator's link state every tick. Nothing is carried in
packet headers, sampled, delayed or lost, so this is an idealized upper bound on
what P4-based In-band Network Telemetry would report.

Links are full-duplex in the simulator; features and labels for each physical
link take the max over its two directions, so the GNN still sees 32 links.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from config import MODEL, SIMULATION
from simulation.network_engine import NetworkEngine, merge_directions

LABEL_MODES = ("offered", "served_or_queued")


def congestion_label(data, capacity, buffer_size, mode=None):
    """1.0 if a (merged) link snapshot counts as congested under `mode`."""
    mode = MODEL["label_mode"] if mode is None else mode
    threshold = MODEL["congestion_threshold"]
    if mode == "offered":
        return float(data["utilization"] > threshold)
    if mode == "served_or_queued":
        return float(data["packets_out"] / capacity > threshold
                     or data["queue_depth"] > MODEL["queue_label_fraction"] * buffer_size)
    raise ValueError(f"unknown label_mode {mode!r}; expected one of {LABEL_MODES}")


class INTReport:
    """A single INT telemetry report for one hop."""

    __slots__ = ["switch_id", "ingress_port", "egress_port",
                 "queue_depth", "timestamp_us", "link_utilization",
                 "packet_drops", "latency_us"]

    def __init__(self, switch_id, ingress_port, egress_port,
                 queue_depth, timestamp_us, link_utilization,
                 packet_drops, latency_us):
        self.switch_id = switch_id
        self.ingress_port = ingress_port
        self.egress_port = egress_port
        self.queue_depth = queue_depth
        self.timestamp_us = timestamp_us
        self.link_utilization = link_utilization
        self.packet_drops = packet_drops
        self.latency_us = latency_us


class TelemetryCollector:
    """
    Collects INT-style telemetry from the network engine
    and converts it into GNN-ready feature matrices.
    """

    def __init__(self, graph: nx.Graph, engine: NetworkEngine):
        self.G = graph
        self.engine = engine
        self.all_switches = graph.graph["all_switches"]
        self.switch_set = set(self.all_switches)

        # Build edge list for switch-to-switch links
        self.switch_edges = []
        seen = set()
        for u, v in graph.edges():
            if u in self.switch_set and v in self.switch_set:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    self.switch_edges.append(key)
                    seen.add(key)

        self.switch_to_idx = {sw: i for i, sw in enumerate(self.all_switches)}
        self.edge_to_idx = {e: i for i, e in enumerate(self.switch_edges)}
        # Also map reversed edges
        for (u, v), idx in list(self.edge_to_idx.items()):
            self.edge_to_idx[(v, u)] = idx

        # Time-series storage
        self.node_features_history = []   # List of [num_switches, node_feat_dim]
        self.edge_features_history = []   # List of [num_edges, edge_feat_dim]
        self.edge_labels_history = []     # List of [num_edges] binary labels
        self.snapshots = []               # Raw snapshot dicts

    def collect_tick(self, snapshot: dict):
        """
        Process one simulation tick's snapshot into feature matrices.

        Node features (per switch):
          [0] avg_queue_occupancy   (normalized by buffer size)
          [1] total_throughput      (normalized by capacity)
          [2] total_drops           (normalized)
          [3] buffer_usage          (fraction 0-1)

        Edge features (per switch-to-switch link):
          [0] link_utilization      (0-1)
          [1] queue_depth           (normalized by buffer size)
          [2] packet_drop_rate      (drops / capacity)
          [3] latency               (normalized, μs / 100)
        """
        self.snapshots.append(snapshot)

        num_switches = len(self.all_switches)
        num_edges = len(self.switch_edges)
        buffer_size = self.G.edges[self.switch_edges[0][0], self.switch_edges[0][1]]["buffer_size"]
        capacity = self.G.edges[self.switch_edges[0][0], self.switch_edges[0][1]]["capacity"]

        # ── Node Features ─────────────────────────────────────────
        node_feats = np.zeros((num_switches, MODEL["node_feature_dim"]), dtype=np.float32)

        for sw, data in snapshot["switches"].items():
            if sw in self.switch_to_idx:
                idx = self.switch_to_idx[sw]
                node_feats[idx, 0] = data["avg_queue"] / max(1, buffer_size)
                node_feats[idx, 1] = data["total_throughput"] / max(1, capacity * self.G.degree(sw))  # normalized
                node_feats[idx, 2] = min(1.0, data["total_drops"] / max(1, capacity))
                node_feats[idx, 3] = data["buffer_usage"]

        # ── Edge Features ─────────────────────────────────────────
        edge_feats = np.zeros((num_edges, MODEL["edge_feature_dim"]), dtype=np.float32)
        edge_labels = np.zeros(num_edges, dtype=np.float32)

        # Max over both directions; a link is congested if either direction is.
        for key, data in merge_directions(snapshot["links"]).items():
            if key in self.edge_to_idx:
                idx = self.edge_to_idx[key]
                edge_feats[idx, 0] = min(1.0, data["utilization"])
                edge_feats[idx, 1] = data["queue_depth"] / max(1, buffer_size)
                edge_feats[idx, 2] = min(1.0, data["packets_dropped"] / max(1, capacity))
                edge_feats[idx, 3] = data["latency_us"] / 100.0  # Normalize
                edge_labels[idx] = congestion_label(data, capacity, buffer_size)

        self.node_features_history.append(node_feats)
        self.edge_features_history.append(edge_feats)
        self.edge_labels_history.append(edge_labels)

    def get_pyg_edge_index(self) -> np.ndarray:
        """
        Get edge_index in COO format for PyTorch Geometric.
        Returns shape [2, num_edges * 2] (bidirectional).
        """
        src_list = []
        dst_list = []
        for u, v in self.switch_edges:
            u_idx = self.switch_to_idx[u]
            v_idx = self.switch_to_idx[v]
            # Bidirectional
            src_list.extend([u_idx, v_idx])
            dst_list.extend([v_idx, u_idx])

        return np.array([src_list, dst_list], dtype=np.int64)

    def get_windowed_samples(self, window_size: int = None,
                             horizon: int = None) -> list:
        """
        Create windowed training samples from the time-series data.

        Each sample:
          - node_features: [window_size, num_switches, node_feat_dim]
          - edge_features: [window_size, num_edges, edge_feat_dim]
          - labels: [num_edges] — congestion at time t + horizon

        Returns
        -------
        list of dicts with 'node_features', 'edge_features', 'labels'
        """
        if window_size is None:
            window_size = MODEL["window_size"]
        if horizon is None:
            horizon = MODEL["prediction_horizon"]

        if window_size < 1 or horizon < 1:
            raise ValueError("window and forecast horizon must be positive")
        total = len(self.node_features_history)
        samples = []

        for t in range(window_size, total - horizon + 1):
            # Window of features
            node_feats = np.stack(
                self.node_features_history[t - window_size:t], axis=0
            )
            edge_feats = np.stack(
                self.edge_features_history[t - window_size:t], axis=0
            )
            # Label: congestion at future time t + horizon
            labels = self.edge_labels_history[t - 1 + horizon]

            samples.append({
                "node_features": node_feats,
                "edge_features": edge_feats,
                "labels": labels,
                "tick": t,
                "feature_start": t - window_size,
                "feature_end": t - 1,
                "target_tick": t - 1 + horizon,
            })

        return samples

    def get_summary(self) -> dict:
        """Get summary statistics of collected telemetry."""
        total_ticks = len(self.node_features_history)
        if total_ticks == 0:
            return {"ticks": 0}

        all_utils = np.array([ef[:, 0] for ef in self.edge_features_history])
        all_labels = np.array(self.edge_labels_history)

        return {
            "ticks": total_ticks,
            "num_switches": len(self.all_switches),
            "num_links": len(self.switch_edges),
            "avg_utilization": float(np.mean(all_utils)),
            "max_utilization": float(np.max(all_utils)),
            "congestion_ratio": float(np.mean(all_labels)),
            "total_congested_events": int(np.sum(all_labels)),
        }

    def reset(self):
        """Clear all collected telemetry."""
        self.node_features_history.clear()
        self.edge_features_history.clear()
        self.edge_labels_history.clear()
        self.snapshots.clear()


def run_simulation_and_collect(graph: nx.Graph, flows: list,
                               total_ticks: int = None) -> TelemetryCollector:
    """
    Convenience function: run the full simulation and return telemetry.
    """
    if total_ticks is None:
        total_ticks = SIMULATION["total_ticks"]

    engine = NetworkEngine(graph)
    engine.set_flows(flows)

    collector = TelemetryCollector(graph, engine)

    for t in range(total_ticks):
        snapshot = engine.step()
        collector.collect_tick(snapshot)

    return collector
