"""
GNN-Based Proactive Congestion Pre-emption in Data Center Fabrics
Using In-Band Telemetry
=================================================================

Entry point for the complete system. Supports three modes:

  python main.py --train     # Generate data + train GNN model
  python main.py --evaluate  # Evaluate model + generate plots
  python main.py --demo      # Launch real-time dashboard
  python main.py --all       # Run everything end-to-end
"""

import argparse
import os
import sys
import time

# Fix Windows console encoding for Unicode
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOPOLOGY, SIMULATION, MODEL, TRAINING, REROUTING,
    MODEL_DIR, DATA_DIR, PLOTS_DIR, CHECKPOINT_OUT, DATA_OUT, EVALUATE_RESULTS_OUT,
    check_output_path, ensure_dirs
)
import json


def generate_data(data_out=DATA_OUT):
    """Phase 1: Build topology, generate traffic, run simulations, collect telemetry."""
    print("\n" + "="*60)
    print("  Phase 1: Data Generation")
    print("="*60)

    from simulation.topology import build_fat_tree, print_topology_summary
    from simulation.traffic_generator import generate_training_scenarios
    from simulation.telemetry import run_simulation_and_collect

    # Build topology
    print("\n[*] Building Fat-Tree topology...")
    G = build_fat_tree()
    print_topology_summary(G)

    # Generate traffic scenarios
    print("[*] Generating traffic scenarios...")
    scenarios = generate_training_scenarios(G, num_scenarios=SIMULATION["num_scenarios"])

    # Run simulations and collect telemetry
    print("\n[*] Running simulations and collecting INT telemetry...")
    all_samples = []
    edge_index = None

    for i, flows in enumerate(scenarios):
        print(f"\n  Scenario {i+1}/{len(scenarios)}: {len(flows)} flows")
        t0 = time.time()

        collector = run_simulation_and_collect(
            G, flows, total_ticks=SIMULATION["total_ticks"]
        )

        if edge_index is None:
            edge_index = collector.get_pyg_edge_index()

        samples = collector.get_windowed_samples()
        for sample in samples:
            sample["scenario_id"] = i
        all_samples.extend(samples)

        summary = collector.get_summary()
        elapsed = time.time() - t0
        print(f"    Ticks: {summary['ticks']}, "
              f"Samples: {len(samples)}, "
              f"Congestion ratio: {summary['congestion_ratio']:.3f}, "
              f"Time: {elapsed:.1f}s")

    print(f"\n  Total training samples: {len(all_samples)}")

    # Save datasets
    from model.dataset import build_datasets, save_datasets

    train_ds, val_ds, test_ds = build_datasets(all_samples, edge_index)
    save_datasets(train_ds, val_ds, test_ds, data_out)

    return G, train_ds, val_ds, test_ds, edge_index


def train(train_ds=None, val_ds=None, edge_index=None, epochs=None,
          checkpoint_out=CHECKPOINT_OUT, data_out=DATA_OUT):
    """Phase 2: Train the GCN-LSTM model."""
    print("\n" + "="*60)
    print("  Phase 2: Model Training")
    print("="*60)

    import torch
    from model.dataset import load_datasets
    from model.gnn_model import CongestionPredictor, load_compatible_state_dict
    from model.train import train_model
    from simulation.topology import build_fat_tree
    from simulation.telemetry import TelemetryCollector
    from simulation.network_engine import NetworkEngine

    # Load datasets if not provided
    if train_ds is None:
        print("\n[*] Loading saved datasets...")
        train_ds, val_ds, test_ds = load_datasets(data_out)
    else:
        _, _, test_ds = load_datasets(data_out)

    # Determine edge pairs for the model
    G = build_fat_tree()
    engine = NetworkEngine(G)
    collector = TelemetryCollector(G, engine)
    edge_pairs = [(collector.switch_to_idx[u], collector.switch_to_idx[v])
                  for u, v in collector.switch_edges]

    if edge_index is None:
        edge_index = collector.get_pyg_edge_index()

    # Build model
    print("\n[*] Building GCN-LSTM model...")
    model = CongestionPredictor(
        num_edges=len(collector.switch_edges),
        edge_pairs=edge_pairs,
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Train
    if epochs is None:
        epochs = TRAINING["epochs"]

    history = train_model(model, train_ds, val_ds, device=device, epochs=epochs,
                          checkpoint_path=checkpoint_out)

    return model, history, test_ds


def evaluate(model=None, test_ds=None, history=None, checkpoint_out=CHECKPOINT_OUT,
             data_out=DATA_OUT, results_out=EVALUATE_RESULTS_OUT):
    """Phase 3: Evaluate model and generate plots."""
    print("\n" + "="*60)
    print("  Phase 3: Evaluation")
    print("="*60)

    import torch
    from model.dataset import load_datasets
    from model.gnn_model import CongestionPredictor, load_compatible_state_dict, check_checkpoint_schema
    from model.evaluate import generate_all_plots
    from simulation.topology import build_fat_tree
    from simulation.telemetry import TelemetryCollector
    from simulation.network_engine import NetworkEngine
    from simulation.traffic_generator import TrafficGenerator
    from routing.comparator import run_comparison

    check_output_path(results_out)
    threshold = (history or {}).get("alert_threshold")
    temperature = (history or {}).get("temperature", 1.0)
    # Load model if not provided
    if model is None:
        print("\n[*] Loading trained model...")
        G = build_fat_tree()
        engine = NetworkEngine(G)
        collector = TelemetryCollector(G, engine)
        edge_pairs = [(collector.switch_to_idx[u], collector.switch_to_idx[v])
                      for u, v in collector.switch_edges]

        model = CongestionPredictor(
            num_edges=len(collector.switch_edges),
            edge_pairs=edge_pairs,
        )

        if os.path.exists(checkpoint_out):
            checkpoint = torch.load(checkpoint_out, map_location="cpu", weights_only=True)
            check_checkpoint_schema(checkpoint)
            load_compatible_state_dict(model, checkpoint["model_state_dict"])
            threshold = checkpoint["alert_threshold"]
            temperature = checkpoint["temperature"]
            print(f"  Loaded checkpoint (epoch {checkpoint['epoch']}, "
                  f"val_ap={checkpoint['val_ap']:.4f}, alert_threshold={threshold:.4f})")
        else:
            raise FileNotFoundError(f"Train first: python main.py --train (no {checkpoint_out})")

    if test_ds is None:
        _, _, test_ds = load_datasets(data_out)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calibrated probabilities: alert_threshold was tuned on temperature-scaled validation outputs.
    from model.calibration import TemperatureScaled
    model = TemperatureScaled(model, temperature).to(device).eval()
    # Run comparison
    print("\n[*] Running routing strategy comparison...")
    G = build_fat_tree()
    gen = TrafficGenerator(G, seed=999)
    test_flows = gen.generate_scenario(total_ticks=200, pattern="hotspot")

    comparison_results = run_comparison(
        G, test_flows, model=model, device=device, total_ticks=200
    )

    # Generate all plots (plots/v3/; the tracked plots/*.png are schema-2 outputs)
    labels, probs, last_labels = generate_all_plots(
        model, test_ds, history,
        comparison_results=comparison_results,
        device=device, threshold=threshold,
    )

    from model.metrics import forecast_metrics
    results = dict(checkpoint=checkpoint_out, label_mode=MODEL["label_mode"], alert_threshold=threshold,
                   temperature=temperature,
                   forecast=forecast_metrics(labels, probs, threshold, last_labels),
                   comparison=comparison_results)
    os.makedirs(os.path.dirname(os.path.abspath(results_out)), exist_ok=True)
    with open(results_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"  Wrote {results_out}")

    return comparison_results


def demo():
    """Phase 4: Launch the real-time dashboard."""
    print("\n" + "="*60)
    print("  Phase 4: Real-Time Dashboard")
    print("="*60)

    from dashboard.server import launch_dashboard
    launch_dashboard()


def main():
    parser = argparse.ArgumentParser(
        description="GNN Congestion Pre-emption System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train              # Generate data + train model
  python main.py --evaluate           # Evaluate + generate plots
  python main.py --demo              # Launch dashboard
  python main.py --all               # Full end-to-end pipeline
  python main.py --train --epochs 50 # Train for 50 epochs
        """
    )
    parser.add_argument("--train", action="store_true",
                        help="Generate data and train the GNN model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model and generate plots")
    parser.add_argument("--demo", action="store_true",
                        help="Launch the real-time dashboard")
    parser.add_argument("--all", action="store_true",
                        help="Run everything: train → evaluate → demo")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--checkpoint-out", default=CHECKPOINT_OUT,
                        help="Checkpoint to write (--train) or load (--evaluate)")
    parser.add_argument("--data-out", default=DATA_OUT,
                        help="Dataset pickle to write (--train) or load (--evaluate)")
    parser.add_argument("--results-out", default=EVALUATE_RESULTS_OUT,
                        help="Evaluation results JSON (--evaluate)")

    args = parser.parse_args()
    for path in (args.checkpoint_out, args.data_out, args.results_out):
        check_output_path(path)

    # If no args, show help
    if not any([args.train, args.evaluate, args.demo, args.all]):
        parser.print_help()
        print("\n[TIP] Quickstart: python main.py --all")
        return

    ensure_dirs()

    print("\n" + "=" * 60)
    print("  GNN-Based Proactive Congestion Pre-emption")
    print("  Data Center Fabrics | In-Band Telemetry")
    print("=" * 60)

    if args.all or args.train:
        G, train_ds, val_ds, test_ds, edge_index = generate_data(args.data_out)
        model, history, test_ds = train(train_ds, val_ds, edge_index, args.epochs,
                                        args.checkpoint_out, args.data_out)

        if args.all or args.evaluate:
            evaluate(model, test_ds, history, args.checkpoint_out, args.data_out, args.results_out)
    elif args.evaluate:
        evaluate(checkpoint_out=args.checkpoint_out, data_out=args.data_out, results_out=args.results_out)

    if args.all or args.demo:
        demo()

    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
