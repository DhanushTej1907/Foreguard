"""
Routing Strategy Comparator
============================
Runs side-by-side simulations comparing:
  1. No rerouting (static ECMP)
  2. Reactive rerouting (reroute after congestion detected)
  3. Proactive rerouting (reroute based on GNN predictions)

Collects and compares: avg latency, max latency, packet drops, throughput.
"""

import copy
import numpy as np
import torch
from simulation.topology import build_fat_tree
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector
from routing.rerouter import ProactiveRerouter, ReactiveRerouter
from config import MODEL, SIMULATION


def run_comparison(graph, flows, model=None, device="cpu",
                   total_ticks=None) -> dict:
    """
    Run all three strategies on the same traffic and compare.

    Parameters
    ----------
    graph : nx.Graph
        The Fat-Tree topology.
    flows : list[Flow]
        The traffic flows to simulate.
    model : CongestionPredictor, optional
        Trained GNN model (needed for proactive strategy).
    device : str
    total_ticks : int

    Returns
    -------
    dict : results for each strategy
    """
    if total_ticks is None:
        total_ticks = min(SIMULATION["total_ticks"], 200)

    results = {}

    # ── Strategy 1: No Rerouting ──────────────────────────────────
    print("\n  Running: No Rerouting (static ECMP)...")
    results["No Rerouting"] = _run_static(graph, flows, total_ticks)

    # ── Strategy 2: Reactive ──────────────────────────────────────
    print("  Running: Reactive Rerouting...")
    results["Reactive"] = _run_reactive(graph, flows, total_ticks)

    # ── Strategy 3: Proactive (GNN) ──────────────────────────────
    if model is not None:
        print("  Running: Proactive Rerouting (GNN)...")
        results["Proactive (GNN)"] = _run_proactive(
            graph, flows, model, device, total_ticks
        )
    else:
        print("  [WARN] Skipping proactive (no trained model provided)")

    # Print comparison
    _print_comparison(results)

    return results


def _deep_copy_flows(flows):
    """Create deep copies of flows with reset state."""
    new_flows = []
    for f in flows:
        nf = copy.copy(f)
        nf.current_path = []
        nf.active = False
        new_flows.append(nf)
    return new_flows


def _collect_metrics(engine, total_ticks) -> dict:
    """Collect summary metrics from a simulation run."""
    return engine.metrics()


def _run_static(graph, flows, total_ticks) -> dict:
    """Run simulation with no rerouting."""
    engine = NetworkEngine(graph)
    sim_flows = _deep_copy_flows(flows)
    engine.set_flows(sim_flows)

    for t in range(total_ticks):
        engine.step()

    return _collect_metrics(engine, total_ticks)


def _run_reactive(graph, flows, total_ticks) -> dict:
    """Run simulation with reactive rerouting."""
    engine = NetworkEngine(graph)
    sim_flows = _deep_copy_flows(flows)
    engine.set_flows(sim_flows)

    rerouter = ReactiveRerouter(graph)

    for t in range(total_ticks):
        snapshot = engine.step()
        # Reactive: check current congestion and reroute
        rerouter.evaluate_and_reroute(sim_flows, engine, t)

    metrics = _collect_metrics(engine, total_ticks)
    stats = rerouter.get_stats()
    metrics["reroutes"] = stats["total_reroutes"]
    return metrics


def _run_proactive(graph, flows, model, device, total_ticks) -> dict:
    """Run simulation with proactive GNN-based rerouting."""
    engine = NetworkEngine(graph)
    sim_flows = _deep_copy_flows(flows)
    engine.set_flows(sim_flows)

    collector = TelemetryCollector(graph, engine)

    # Build edge mapping for the rerouter
    edge_to_idx = collector.edge_to_idx
    edge_index = collector.get_pyg_edge_index()
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)

    rerouter = ProactiveRerouter(graph, edge_to_idx)

    model.to(device)
    model.eval()
    window = MODEL["window_size"]

    for t in range(total_ticks):
        snapshot = engine.step()
        collector.collect_tick(snapshot)

        # Only predict once we have enough history
        if t + 1 >= window:
            # Build input tensors from last `window` timesteps
            node_feats = np.stack(
                collector.node_features_history[-window:], axis=0
            )  # [W, N, F_node]
            edge_feats = np.stack(
                collector.edge_features_history[-window:], axis=0
            )  # [W, E, F_edge]

            # Add batch dimension
            node_tensor = torch.tensor(
                node_feats, dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, W, N, F_node]
            edge_tensor = torch.tensor(
                edge_feats, dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, W, E, F_edge]

            with torch.no_grad():
                predictions = model(node_tensor, edge_tensor, edge_index_tensor)
                predictions = predictions.squeeze(0).cpu().numpy()  # [E]

            # Proactive reroute based on predictions
            rerouter.evaluate_and_reroute(sim_flows, predictions, engine, t)

    metrics = _collect_metrics(engine, total_ticks)
    stats = rerouter.get_stats()
    metrics["reroutes"] = stats["total_reroutes"]
    return metrics


def _print_comparison(results: dict):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"  Routing Strategy Comparison")
    print(f"{'='*70}")
    print(f"  {'Strategy':<22} {'Avg Lat(μs)':>12} {'Max Lat(μs)':>12} "
          f"{'Drops':>8} {'Throughput':>11}")
    print(f"  {'-'*65}")

    for name, m in results.items():
        print(f"  {name:<22} {m['avg_latency']:>12.2f} {m['max_latency']:>12.2f} "
              f"{m['total_drops']:>8,} {m['avg_throughput']:>11.1f}")

    print(f"{'='*70}")

    # Compute improvements
    if "No Rerouting" in results and "Proactive (GNN)" in results:
        base = results["No Rerouting"]
        pro = results["Proactive (GNN)"]

        if base["total_drops"] > 0:
            drop_reduction = (1 - pro["total_drops"] / base["total_drops"]) * 100
            print(f"\n  [>>] Proactive vs No-Rerouting:")
            print(f"     Packet drops reduced by {drop_reduction:.1f}%")

        if base["avg_latency"] > 0:
            lat_change = (pro["avg_latency"] / base["avg_latency"] - 1) * 100
            if lat_change <= 0:
                print(f"     Avg link queue-delay proxy reduced by {-lat_change:.1f}%")
            else:
                print(f"     Avg link queue-delay proxy increased by {lat_change:.1f}%")
