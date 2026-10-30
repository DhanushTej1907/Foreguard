"""
Proactive Rerouting Engine
===========================
Takes GNN congestion predictions and proactively reroutes flows
before congestion materializes.

Strategy:
  1. For each link, check if predicted congestion probability > threshold
  2. For affected flows, compute K-shortest alternative paths
  3. Assign weights inversely proportional to predicted congestion
  4. Redistribute traffic across available paths
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from config import REROUTING, TOPOLOGY


class ProactiveRerouter:
    """
    Reroutes flows proactively based on GNN congestion predictions.
    """

    def __init__(self, graph: nx.Graph, edge_to_idx: dict):
        self.G = graph
        self.edge_to_idx = edge_to_idx
        self.all_switches = set(graph.graph["all_switches"])
        self.reroute_log = []  # Log of rerouting decisions
        self.last_reroute_tick = {}  # flow_id → last reroute tick

    def evaluate_and_reroute(self, flows: list, predictions: np.ndarray,
                              engine, tick: int) -> list:
        """
        Evaluate GNN predictions and reroute flows if needed.

        Parameters
        ----------
        flows : list[Flow]
            Active flows in the simulation.
        predictions : np.ndarray
            Per-edge congestion probabilities from the GNN, shape [num_edges].
        engine : NetworkEngine
            The simulation engine (to update paths).
        tick : int
            Current simulation tick.

        Returns
        -------
        list : rerouting events for the dashboard
        """
        threshold = REROUTING["congestion_alert_threshold"]
        cooldown = REROUTING["reroute_cooldown_ticks"]

        # Find at-risk edges
        at_risk_edges = set()
        for (u, v), idx in self.edge_to_idx.items():
            if idx < len(predictions) and predictions[idx] > threshold:
                at_risk_edges.add((u, v))

        if not at_risk_edges:
            return []

        events = []

        for flow in flows:
            if not flow.active:
                continue

            # Check cooldown
            if flow.flow_id in self.last_reroute_tick:
                if tick - self.last_reroute_tick[flow.flow_id] < cooldown:
                    continue

            # Check if current path uses any at-risk edge
            path = flow.current_path
            path_at_risk = False
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                rev_edge = (path[i+1], path[i])
                if edge in at_risk_edges or rev_edge in at_risk_edges:
                    path_at_risk = True
                    break

            if not path_at_risk:
                continue

            # Find alternative path avoiding at-risk links
            new_path = self._find_safe_path(
                flow.src, flow.dst, predictions, at_risk_edges
            )

            if new_path and new_path != path:
                old_path = flow.current_path
                flow.current_path = new_path
                self.last_reroute_tick[flow.flow_id] = tick

                event = {
                    "tick": tick,
                    "flow_id": flow.flow_id,
                    "flow_type": flow.flow_type,
                    "src": flow.src,
                    "dst": flow.dst,
                    "old_path": old_path,
                    "new_path": new_path,
                    "reason": "proactive",
                }
                self.reroute_log.append(event)
                events.append(event)

        return events

    def _find_safe_path(self, src: str, dst: str,
                         predictions: np.ndarray,
                         at_risk_edges: set) -> list:
        """
        Find a path that avoids or minimizes traversal of at-risk edges.
        Uses Dijkstra with weights based on predicted congestion.
        """
        # Create a weighted copy of the graph
        G_weighted = self.G.copy()

        for u, v in G_weighted.edges():
            key = (min(u, v), max(u, v))
            rev_key = (max(u, v), min(u, v))

            # Base weight = 1
            weight = 1.0

            # If edge has a prediction, use predicted congestion as penalty
            if key in self.edge_to_idx:
                idx = self.edge_to_idx[key]
                if idx < len(predictions):
                    congestion_prob = predictions[idx]
                    # Higher predicted congestion → much higher weight
                    weight = 1.0 + congestion_prob * 10.0

            # Heavily penalize at-risk edges
            if (u, v) in at_risk_edges or (v, u) in at_risk_edges:
                weight += 50.0

            G_weighted[u][v]["weight"] = weight

        try:
            path = nx.shortest_path(G_weighted, src, dst, weight="weight")
            return path
        except nx.NetworkXNoPath:
            return []

    def get_stats(self) -> dict:
        """Get rerouting statistics."""
        return {
            "total_reroutes": len(self.reroute_log),
            "proactive_reroutes": sum(1 for e in self.reroute_log
                                      if e["reason"] == "proactive"),
        }


class ReactiveRerouter:
    """
    Baseline: reroutes flows only AFTER congestion is detected.
    """

    def __init__(self, graph: nx.Graph):
        self.G = graph
        self.all_switches = set(graph.graph["all_switches"])
        self.reroute_log = []
        self.last_reroute_tick = {}

    def evaluate_and_reroute(self, flows: list, engine, tick: int) -> list:
        """
        Reroute based on current (not predicted) congestion.
        """
        threshold = REROUTING["reactive_threshold"]
        cooldown = REROUTING["reroute_cooldown_ticks"]

        # Find currently congested directed switch-to-switch links. Host links
        # have no alternative path, so they are never a reason to reroute.
        congested_edges = set()
        for (u, v), state in engine.link_states.items():
            if u in self.all_switches and v in self.all_switches and state.utilization > threshold:
                congested_edges.add((u, v))

        if not congested_edges:
            return []

        events = []

        for flow in flows:
            if not flow.active:
                continue

            if flow.flow_id in self.last_reroute_tick:
                if tick - self.last_reroute_tick[flow.flow_id] < cooldown:
                    continue

            # Check if current path uses congested edge
            path = flow.current_path
            path_congested = False
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) in congested_edges:
                    path_congested = True
                    break

            if not path_congested:
                continue

            # Find alternative avoiding congested links
            new_path = self._find_uncongested_path(
                flow.src, flow.dst, congested_edges
            )

            if new_path and new_path != path:
                old_path = flow.current_path
                flow.current_path = new_path
                self.last_reroute_tick[flow.flow_id] = tick

                event = {
                    "tick": tick,
                    "flow_id": flow.flow_id,
                    "src": flow.src,
                    "dst": flow.dst,
                    "old_path": old_path,
                    "new_path": new_path,
                    "reason": "reactive",
                }
                self.reroute_log.append(event)
                events.append(event)

        return events

    def _find_uncongested_path(self, src, dst, congested_edges):
        """Find path avoiding currently congested edges (direction-aware)."""
        # NetworkX calls weight(u, v, d) in the direction of traversal.
        weight = lambda u, v, d: 100.0 if (u, v) in congested_edges else 1.0
        try:
            return nx.shortest_path(self.G, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def get_stats(self):
        return {
            "total_reroutes": len(self.reroute_log),
            "reactive_reroutes": sum(1 for e in self.reroute_log
                                      if e["reason"] == "reactive"),
        }
