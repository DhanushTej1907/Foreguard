"""
Dashboard Server
=================
FastAPI backend with WebSocket support for the real-time visualization dashboard.

Endpoints:
  GET /           → Serves the dashboard HTML
  WS  /ws         → WebSocket for live simulation data
  POST /api/start → Start simulation
  POST /api/stop  → Stop simulation
  POST /api/spike → Inject traffic spike
  POST /api/config → Update GNN threshold
"""

import os
import json
import asyncio
import time
import numpy as np
import torch
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from contextlib import asynccontextmanager, suppress
from pydantic import BaseModel, Field
from routing.guarded import GuardedRerouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import DASHBOARD, MODEL, TOPOLOGY, REROUTING, ensure_dirs
from simulation.topology import build_fat_tree, get_switch_index_map
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine, merge_directions
from simulation.telemetry import TelemetryCollector
from routing.rerouter import ProactiveRerouter, ReactiveRerouter

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

app = FastAPI(lifespan=lifespan, title="NeuralFabric — GNN Congestion Pre-emption Engine")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state
sim_state = {
    "running": False,
    "graph": None,
    "engine": None,
    "collector": None,
    "flows": [],
    "model": None,
    "device": "cpu",
    "tick": 0,
    "speed": 1.0,
    "threshold": REROUTING["congestion_alert_threshold"],
    "rerouter": None,
    "reactive_rerouter": None,
    "connected_clients": set(),
    "total_reroutes": 0,
}


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/start")
async def start_simulation():
    """Initialize and start the simulation."""
    if sim_state["running"]:
        return JSONResponse({"status": "already_running"})

    # Build topology
    G = build_fat_tree()
    sim_state["graph"] = G
    sim_state["engine"] = NetworkEngine(G)

    # Generate traffic
    gen = TrafficGenerator(G, seed=int(time.time()) % 10000)
    flows = gen.generate_scenario(total_ticks=2000, pattern="mixed")
    sim_state["flows"] = flows
    sim_state["engine"].set_flows(flows)

    # Telemetry collector
    sim_state["collector"] = TelemetryCollector(G, sim_state["engine"])

    # Initialize rerouters
    collector = sim_state["collector"]
    sim_state["reactive_rerouter"] = ReactiveRerouter(G)

    sim_state["model"] = None
    sim_state["rerouter"] = None
    # Try loading trained model
    try:
        from model.gnn_model import CongestionPredictor, load_compatible_state_dict, check_checkpoint_schema
        from config import CHECKPOINT_OUT

        checkpoint_path = CHECKPOINT_OUT
        if os.path.exists(checkpoint_path):
            edge_pairs = [(collector.switch_to_idx[u], collector.switch_to_idx[v])
                          for u, v in collector.switch_edges]

            model = CongestionPredictor(
                num_edges=len(collector.switch_edges),
                edge_pairs=edge_pairs,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            check_checkpoint_schema(checkpoint)
            load_compatible_state_dict(model, checkpoint["model_state_dict"])
            model.eval()
            sim_state["model"] = model

            # Guarded rerouter alerts at the validation-tuned threshold stored in the checkpoint.
            sim_state["threshold"] = checkpoint["alert_threshold"]
            sim_state["rerouter"] = GuardedRerouter(G, collector.edge_to_idx, threshold=sim_state["threshold"])

            print("  [OK] Loaded trained GNN model for live predictions")
        else:
            print("  [WARN] No trained model found, running without GNN predictions")
    except Exception as e:
        print(f"  [WARN] Could not load model: {e}")

    sim_state["tick"] = 0
    sim_state["total_reroutes"] = 0
    sim_state["running"] = True

    return JSONResponse({"status": "started", "topology": _get_topology_data(G)})


@app.post("/api/stop")
async def stop_simulation():
    sim_state["running"] = False
    return JSONResponse({"status": "stopped"})


@app.post("/api/spike")
async def inject_spike():
    """Inject a traffic hotspot spike."""
    if not sim_state["running"]:
        return JSONResponse({"status": "not_running"})

    G = sim_state["graph"]
    gen = TrafficGenerator(G, seed=int(time.time()) % 10000)
    spike_flows = gen.generate_scenario(total_ticks=100, pattern="hotspot")

    # Adjust start times relative to current tick
    next_id = max((f.flow_id for f in sim_state["flows"]), default=0)
    for offset, f in enumerate(spike_flows, 1):
        f.flow_id = next_id + offset
        f.start_tick += sim_state["tick"]
        if f.flow_type == "incast":
            f.start_tick = sim_state["tick"] + 5
            f.duration = 40

    sim_state["flows"].extend(spike_flows)
    sim_state["engine"].set_flows(sim_state["flows"])

    return JSONResponse({"status": "spike_injected", "new_flows": len(spike_flows)})


class ConfigUpdate(BaseModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    speed: float | None = Field(default=None, ge=.1, le=20)


@app.post("/api/config")
async def update_config(data: ConfigUpdate):
    for key, value in data.model_dump(exclude_none=True).items():
        sim_state[key] = value
    return {"status": "updated"}


@app.get("/api/status")
async def status():
    return {"running": sim_state["running"], "tick": sim_state["tick"],
            "model_loaded": sim_state["model"] is not None,
            "controller": "guarded GNN" if sim_state["model"] is not None else "static ECMP",
            "scope": "synthetic full-duplex simulator; no PFC or RDMA"}


async def simulation_loop():
    """One producer per server, independent of the number of viewers."""
    while True:
        if not sim_state["running"]:
            await asyncio.sleep(.05)
            continue
        try:
            engine, collector = sim_state["engine"], sim_state["collector"]
            snapshot = engine.step()
            collector.collect_tick(snapshot)
            sim_state["tick"] = engine.tick
            predictions, events = None, []
            if sim_state["model"] is not None and len(collector.node_features_history) >= MODEL["window_size"]:
                predictions = _get_predictions(collector, sim_state["model"])
                sim_state["rerouter"].threshold = sim_state["threshold"]
                events = sim_state["rerouter"].evaluate_and_reroute(
                    sim_state["flows"], predictions, engine, engine.tick)
                sim_state["total_reroutes"] += len(events)
            # Bound live telemetry history; offline collectors keep full scenarios.
            for name in ["node_features_history", "edge_features_history", "edge_labels_history",
                         "edge_scores_history", "snapshots"]:
                history = getattr(collector, name)
                del history[:-MODEL["window_size"]]
            msg = _build_update_message(snapshot, predictions, events)
            async def send(client):
                try:
                    await asyncio.wait_for(client.send_json(msg), timeout=.5)
                except Exception:
                    sim_state["connected_clients"].discard(client)
            await asyncio.gather(*(send(c) for c in list(sim_state["connected_clients"])))
        except Exception as exc:
            sim_state["running"] = False
            print(f"Simulation stopped: {exc}")
        await asyncio.sleep(DASHBOARD["ws_update_interval_ms"] / 1000 / sim_state["speed"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_state["connected_clients"].add(websocket)
    try:
        if sim_state["graph"] is not None:
            await websocket.send_json({"type": "topology", "data": _get_topology_data(sim_state["graph"])})
        while True:
            cmd = await websocket.receive_json()
            if cmd.get("action") == "spike":
                await inject_spike()
            elif cmd.get("action") == "toggle":
                if sim_state["engine"] is None:
                    await start_simulation()
                else:
                    sim_state["running"] = not sim_state["running"]
            elif cmd.get("action") == "threshold":
                try:
                    config = ConfigUpdate(threshold=cmd.get("value"))
                    await update_config(config)
                except (ValueError, TypeError):
                    await websocket.send_json({"type": "error", "message": "Threshold must be between 0 and 1"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sim_state["connected_clients"].discard(websocket)


def _get_topology_data(G) -> dict:
    """Convert topology to JSON-serializable format."""
    nodes = []
    for node_id in G.nodes():
        data = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "type": data.get("type", "unknown"),
            "layer": data.get("layer", -1),
            "pod": data.get("pod", -1),
        })

    links = []
    for u, v, data in G.edges(data=True):
        links.append({
            "source": u,
            "target": v,
            "capacity": data.get("capacity", 0),
            "is_host_link": data.get("is_host_link", False),
        })

    return {"nodes": nodes, "links": links}


def _get_predictions(collector, model) -> np.ndarray:
    """Run GNN inference on current telemetry."""
    window = MODEL["window_size"]

    node_feats = np.stack(
        collector.node_features_history[-window:], axis=0
    )
    edge_feats = np.stack(
        collector.edge_features_history[-window:], axis=0
    )
    edge_index = collector.get_pyg_edge_index()

    node_tensor = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0)
    edge_tensor = torch.tensor(edge_feats, dtype=torch.float32).unsqueeze(0)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    with torch.no_grad():
        pred = model(node_tensor, edge_tensor, edge_index_tensor)

    return pred.squeeze(0).numpy()


def _build_update_message(snapshot, predictions=None, reroute_events=None) -> dict:
    """Build WebSocket update message."""
    # Link data: one entry per physical link (max over both directions);
    # the frontend maps u--v and v--u to the same line.
    links = {}
    for (u, v), data in merge_directions(snapshot["links"]).items():
        link_id = f"{u}--{v}"
        links[link_id] = {
            "utilization": round(data["utilization"], 4),
            "queue_depth": data["queue_depth"],
            "drops": data["packets_dropped"],
            "latency": round(data["latency_us"], 2),
        }

    # Predictions
    pred_data = {}
    if predictions is not None:
        for i, prob in enumerate(predictions):
            pred_data[str(i)] = round(float(prob), 4)

    # Global metrics
    metrics = {
        "tick": snapshot["tick"],
        "active_flows": snapshot["active_flows"],
        "total_packets": snapshot["total_packets"],
        "total_drops": snapshot["total_drops"],
        "drop_rate": round(snapshot["total_drops"] / max(1, snapshot["total_packets"]) * 100, 3),
    }

    # Reroute events
    reroute_data = []
    if reroute_events:
        for evt in reroute_events:
            reroute_data.append({
                "flow_id": evt["flow_id"],
                "src": evt["src"],
                "dst": evt["dst"],
                "reason": evt.get("reason", "proactive"),
            })

    return {
        "type": "update",
        "links": links,
        "predictions": pred_data,
        "metrics": metrics,
        "reroute_events": reroute_data,
    }


def launch_dashboard(host=None, port=None):
    """Launch the dashboard server."""
    import uvicorn
    torch.set_num_threads(2)

    host = host or DASHBOARD["host"]
    port = port or DASHBOARD["port"]

    ensure_dirs()

    print(f"\n{'='*60}")
    print(f"  NeuralFabric Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")
