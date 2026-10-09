"""
Dashboard Server
=================
FastAPI backend with WebSocket support for the real-time visualization dashboard.

Endpoints:
  GET  /               → Serves the dashboard HTML
  WS   /ws             → WebSocket for live simulation data
  POST /api/start      → Start simulation
  POST /api/stop       → Stop simulation
  POST /api/spike      → Inject traffic spike
  POST /api/config     → Update GNN alert threshold / speed
  POST /api/controller → Switch routing controller (restores ECMP paths)
  GET  /api/status     → Running state, controller, thresholds
"""

import os
import asyncio
import time
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from contextlib import asynccontextmanager, suppress
from pydantic import BaseModel, Field
from routing.guarded import GuardedRerouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import DASHBOARD, MODEL, REROUTING, ensure_dirs
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine, merge_directions
from simulation.telemetry import TelemetryCollector, congestion_score

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

# Live-switchable routing controllers. Every switch restores ECMP paths, so each
# controller starts from the same routing state.
CONTROLLERS = {
    "ecmp": "ECMP (no rerouting)",
    "guarded_gnn": "Guarded GNN",
    "guarded_score": "Guarded score (no model)",
}
# guarded_score feeds the guard the current served_or_queued congestion score and alerts at
# the label threshold (the benchmark used continuous persistence's val-tuned 0.851).
SCORE_THRESHOLD = MODEL["congestion_threshold"]

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
    "threshold": REROUTING["congestion_alert_threshold"],  # live GNN alert threshold (slider)
    "alert_threshold": None,                                 # calibrated threshold from the checkpoint
    "controller": "ecmp",
    "rerouter": None,
    "connected_clients": set(),
    "total_reroutes": 0,
    "switch_reroutes": 0,
    "switch_baseline": None,
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
    collector = sim_state["collector"]

    sim_state["model"] = None
    sim_state["alert_threshold"] = None
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
            from model.calibration import TemperatureScaled
            # alert_threshold was tuned on temperature-scaled validation probabilities.
            sim_state["model"] = TemperatureScaled(model, checkpoint["temperature"]).eval()

            # The live threshold (slider) starts at the validation-tuned threshold in the checkpoint.
            sim_state["alert_threshold"] = float(checkpoint["alert_threshold"])
            sim_state["threshold"] = sim_state["alert_threshold"]

            print("  [OK] Loaded trained GNN model for live predictions")
        else:
            print("  [WARN] No trained model found, running without GNN predictions")
    except Exception as e:
        print(f"  [WARN] Could not load model: {e}")

    sim_state["tick"] = 0
    sim_state["total_reroutes"] = 0
    _switch_controller("guarded_gnn" if sim_state["model"] is not None else "ecmp")
    sim_state["running"] = True

    return JSONResponse({"status": "started", "topology": _get_topology_data(G), **_controller_status()})


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


class ControllerUpdate(BaseModel):
    controller: str


@app.post("/api/controller")
async def set_controller(data: ControllerUpdate):
    """Switch the routing controller live; ECMP paths are restored and counters reset."""
    if data.controller not in CONTROLLERS:
        raise HTTPException(422, f"unknown controller {data.controller!r}; expected one of {list(CONTROLLERS)}")
    if sim_state["engine"] is None:
        raise HTTPException(409, "start the simulation first")
    if data.controller == "guarded_gnn" and sim_state["model"] is None:
        raise HTTPException(409, "guarded_gnn needs a loaded schema-3 checkpoint")
    _switch_controller(data.controller)
    return {"status": "switched", **_controller_status()}


@app.get("/api/status")
async def status():
    model_loaded = sim_state["model"] is not None
    return {"running": sim_state["running"], "tick": sim_state["tick"],
            **_controller_status(),
            "controllers": {name: {"label": label, "available": name != "guarded_gnn" or model_loaded}
                            for name, label in CONTROLLERS.items()},
            "scope": "synthetic full-duplex simulator; no PFC or RDMA"}


def _switch_controller(name):
    """Activate a controller: restore every flow's ECMP path, start a fresh guard (no
    cooldowns carried over) and reset the since-switch counters."""
    engine, collector = sim_state["engine"], sim_state["collector"]
    for f in sim_state["flows"]:
        f.current_path = engine.get_ecmp_path(f)
    if name == "ecmp":
        sim_state["rerouter"] = None
    else:
        threshold = sim_state["threshold"] if name == "guarded_gnn" else SCORE_THRESHOLD
        sim_state["rerouter"] = GuardedRerouter(sim_state["graph"], collector.edge_to_idx, threshold=threshold)
    sim_state["controller"] = name
    sim_state["switch_reroutes"] = 0
    sim_state["switch_baseline"] = dict(tick=engine.tick, total_drops=engine.total_drops,
                                        fabric_drops=engine.fabric_link_drops,
                                        host_drops=engine.host_link_drops)


def _controller_status() -> dict:
    """Active controller, thresholds and cumulative counters since the last switch."""
    engine, base = sim_state["engine"], sim_state["switch_baseline"]
    since = None
    if engine is not None and base is not None:
        since = dict(ticks=engine.tick - base["tick"],
                     total_drops=engine.total_drops - base["total_drops"],
                     fabric_drops=engine.fabric_link_drops - base["fabric_drops"],
                     host_drops=engine.host_link_drops - base["host_drops"],
                     reroutes=sim_state["switch_reroutes"])
    return {"controller": sim_state["controller"],
            "controller_label": CONTROLLERS[sim_state["controller"]],
            "since_switch": since,
            "model_loaded": sim_state["model"] is not None,
            "threshold": sim_state["threshold"],
            "alert_threshold": sim_state["alert_threshold"],
            "score_threshold": SCORE_THRESHOLD}


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
            ready = len(collector.node_features_history) >= MODEL["window_size"]
            # The forecast overlay is shown under every controller; only guarded_gnn routes on it.
            predictions = None
            if ready and sim_state["model"] is not None:
                predictions = _get_predictions(collector, sim_state["model"])
            events, rerouter = [], sim_state["rerouter"]
            if ready and rerouter is not None:
                if sim_state["controller"] == "guarded_gnn":
                    rerouter.threshold = sim_state["threshold"]
                    signal = predictions
                else:
                    signal = collector.edge_scores_history[-1]
                events = rerouter.evaluate_and_reroute(sim_state["flows"], signal, engine, engine.tick)
                sim_state["total_reroutes"] += len(events)
                sim_state["switch_reroutes"] += len(events)
            # Bound live telemetry history; offline collectors keep full scenarios.
            for name in ["node_features_history", "edge_features_history", "edge_labels_history",
                         "edge_scores_history", "snapshots"]:
                history = getattr(collector, name)
                del history[:-MODEL["window_size"]]
            msg = _build_update_message(snapshot, collector, predictions, events)
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
            elif cmd.get("action") == "controller":
                try:
                    await set_controller(ControllerUpdate(controller=cmd.get("value")))
                except ValueError:
                    await websocket.send_json({"type": "error", "message": "Controller must be a string"})
                except HTTPException as exc:
                    await websocket.send_json({"type": "error", "message": exc.detail})
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


def _link_id(u, v) -> str:
    """Physical link ID "u--v" with u < v, as merge_directions and collector.switch_edges order them."""
    u, v = sorted((u, v))
    return f"{u}--{v}"


def _build_update_message(snapshot, collector, predictions=None, reroute_events=None) -> dict:
    """Build WebSocket update message."""
    # Link data: one entry per physical link (max over both directions);
    # the frontend maps u--v and v--u to the same line.
    G = collector.G
    links = {}
    for (u, v), data in merge_directions(snapshot["links"]).items():
        edge = G.edges[u, v]
        # Current congestion is the training label (served_or_queued score > T), not utilization.
        score = congestion_score(data, edge["capacity"], edge["buffer_size"])
        links[_link_id(u, v)] = {
            "utilization": round(data["utilization"], 4),
            "queue_depth": data["queue_depth"],
            "drops": data["packets_dropped"],
            "latency": round(data["latency_us"], 2),
            "score": round(score, 4),
            "congested": score > MODEL["congestion_threshold"],
        }

    # Calibrated P(congested at t + horizon) per fabric link, keyed in collector.switch_edges order.
    pred_data = {}
    if predictions is not None:
        pred_data = {_link_id(u, v): round(float(prob), 4)
                     for (u, v), prob in zip(collector.switch_edges, predictions, strict=True)}

    # Global metrics
    metrics = {
        "tick": snapshot["tick"],
        "active_flows": snapshot["active_flows"],
        "total_packets": snapshot["total_packets"],
        "total_drops": snapshot["total_drops"],
        "drop_rate": round(snapshot["total_drops"] / max(1, snapshot["total_packets"]) * 100, 3),
        "total_reroutes": sim_state["total_reroutes"],
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
                "controller": sim_state["controller"],
                "new_path": list(evt["new_path"]),
            })

    return {
        "type": "update",
        "links": links,
        "predictions": pred_data,
        "metrics": metrics,
        "reroute_events": reroute_data,
        **_controller_status(),
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
