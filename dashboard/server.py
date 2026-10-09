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
  POST /api/workload   → Restart the simulation with another traffic family
  GET  /api/status     → Running state, workload, controller, thresholds
"""

import os
import asyncio
import itertools
import random
import time
from collections import deque
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
from research_benchmark import PATTERNS as WORKLOADS

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
# Live traffic families are the benchmark's; stride shows the guard's effect most clearly
# (ECMP drops on fabric links, every guarded strategy removes them).
DEFAULT_WORKLOAD = "stride"
# Live traffic is played as back-to-back episodes of the benchmark's run length, so each family
# has the load it had in the paper. (Flow start times scale with scenario length: one long
# 2000-tick scenario ramped up so slowly that stride had no drops for its first 300+ ticks.)
EPISODE_TICKS = 150
SPIKE_ID_BASE = 1_000_000  # spike flow IDs; episode flows keep the generator's small IDs
# Event Timeline: congestion / prediction events fire on transitions only, one coalesced event per
# type per tick. A link leaves the congested (or predicted) set only after CLEAR_HOLD_TICKS
# consecutive ticks below its threshold, so links hovering at the threshold do not flood the log
# (mixed traffic: 167 -> 81 events per 150 ticks).
CLEAR_HOLD_TICKS = 5
RECENT_EVENTS = 200  # backlog sent to a viewer that connects mid-run
_event_ids = itertools.count(1)


class _Crossings:
    """Links above a threshold, with hysteresis: a link enters on its first tick above and
    leaves after `hold` consecutive ticks below. update() returns (entered, left)."""

    def __init__(self, hold=CLEAR_HOLD_TICKS):
        self.hold, self.active, self.quiet = hold, set(), {}

    def update(self, above):
        entered = set(above) - self.active
        for link in above:
            self.quiet.pop(link, None)
        left = set()
        for link in self.active - set(above):
            self.quiet[link] = self.quiet.get(link, 0) + 1
            if self.quiet[link] >= self.hold:
                left.add(link)
                del self.quiet[link]
        self.active = (self.active | entered) - left
        return entered, left


# Global state
sim_state = {
    "running": False,
    "workload": DEFAULT_WORKLOAD,
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
    "pending_events": [],          # timeline events, sent with the next update message
    "recent_events": deque(maxlen=RECENT_EVENTS),  # this run's events, replayed to new viewers
    "congestion_tracker": _Crossings(),
    "prediction_tracker": _Crossings(),
}


def _emit(event_type, **fields):
    """Queue a timeline event (sent to every viewer with the next update)."""
    engine = sim_state["engine"]
    event = dict(id=next(_event_ids), type=event_type, tick=engine.tick if engine is not None else 0, **fields)
    sim_state["pending_events"].append(event)
    sim_state["recent_events"].append(event)
    return event


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/start")
async def start_simulation():
    """Initialize and start the simulation with the selected workload."""
    if sim_state["running"]:
        return JSONResponse({"status": "already_running"})
    _start(sim_state["workload"], keep_settings=False)
    _emit("start", workload=sim_state["workload"], controller=sim_state["controller"],
          controller_label=CONTROLLERS[sim_state["controller"]])
    return JSONResponse({"status": "started", "topology": _get_topology_data(sim_state["graph"]), **_run_status()})


def _start(workload, keep_settings):
    """Build and run a fresh simulation of `workload`. A fresh start resets the controller to
    guarded_gnn (ECMP without a model) and the threshold to the checkpoint's calibrated value;
    keep_settings keeps the selected controller (if still available) and the live threshold."""
    controller, threshold = sim_state["controller"], sim_state["threshold"]

    # Build topology
    G = build_fat_tree()
    sim_state["graph"] = G
    sim_state["engine"] = NetworkEngine(G)

    # Generate traffic: the first benchmark-length episode; the loop appends the next ones.
    sim_state["workload"] = workload
    sim_state["flows"] = []
    sim_state["rerouter"] = None
    sim_state["next_spike_id"] = SPIKE_ID_BASE
    sim_state["traffic_rng"] = random.Random(time.time_ns())
    sim_state["congestion_tracker"] = _Crossings()
    sim_state["prediction_tracker"] = _Crossings()
    sim_state["recent_events"].clear()  # the backlog covers the current run only
    _next_episode(0)

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

    if keep_settings:
        sim_state["threshold"] = threshold
        if controller == "guarded_gnn" and sim_state["model"] is None:
            controller = "ecmp"
    else:
        controller = "guarded_gnn" if sim_state["model"] is not None else "ecmp"
    sim_state["tick"] = 0
    sim_state["total_reroutes"] = 0
    _switch_controller(controller)
    sim_state["running"] = True


@app.post("/api/stop")
async def stop_simulation():
    sim_state["running"] = False
    return JSONResponse({"status": "stopped"})


@app.post("/api/spike")
async def inject_spike():
    """Inject a traffic hotspot spike."""
    if not sim_state["running"]:
        return JSONResponse({"status": "not_running"})

    gen = TrafficGenerator(sim_state["graph"], seed=sim_state["traffic_rng"].randrange(2**31))
    spike_flows = gen.generate_scenario(total_ticks=100, pattern="hotspot")
    for f in spike_flows:
        f.flow_id = sim_state["next_spike_id"]
        sim_state["next_spike_id"] += 1
        if f.flow_type == "incast":
            f.start_tick = 5
            f.duration = 40
    _add_flows(spike_flows, start_tick=sim_state["tick"])
    _emit("spike", flows=len(spike_flows))

    return JSONResponse({"status": "spike_injected", "new_flows": len(spike_flows)})


def _add_flows(flows, start_tick):
    """Shift `flows` to start at start_tick and add them to the running engine."""
    for f in flows:
        f.start_tick += start_tick
    sim_state["flows"].extend(flows)
    sim_state["engine"].set_flows(sim_state["flows"])


def _next_episode(start_tick):
    """Append one EPISODE_TICKS-long scenario of the current workload starting at start_tick,
    and drop flows that have finished (their packets in flight stay in the engine's queues).

    Episode flows keep the generator's IDs: ECMP hashes (src, dst, flow_id), so renumbering
    would change which flows collide (stride's ECMP drops vanish with IDs 0..15 instead of 1..16).
    IDs stay unique because the previous episode's flows have all ended by now; spike flows use
    IDs from SPIKE_ID_BASE up."""
    now = sim_state["engine"].tick
    sim_state["flows"] = [f for f in sim_state["flows"] if f.start_tick + f.duration > now]
    live = {f.flow_id for f in sim_state["flows"]}
    if sim_state["rerouter"] is not None:  # no cooldown carried over to a reused ID
        cooldowns = sim_state["rerouter"].last_reroute_tick
        sim_state["rerouter"].last_reroute_tick = {i: t for i, t in cooldowns.items() if i in live}
    sim_state["episode_seed"] = sim_state["traffic_rng"].randrange(2**31)
    gen = TrafficGenerator(sim_state["graph"], seed=sim_state["episode_seed"])
    flows = gen.generate_scenario(total_ticks=EPISODE_TICKS, pattern=sim_state["workload"])
    _add_flows(flows, start_tick)
    # The next episode starts once this one's flows have all ended.
    sim_state["episode_end"] = max((f.start_tick + f.duration for f in flows), default=start_tick + EPISODE_TICKS)


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
    _emit("controller", controller=data.controller, controller_label=CONTROLLERS[data.controller])
    return {"status": "switched", **_run_status()}


class WorkloadUpdate(BaseModel):
    workload: str


@app.post("/api/workload")
async def set_workload(data: WorkloadUpdate):
    """Select a traffic family. Once the simulation exists, switching restarts it with the new
    workload, keeping the controller and the live threshold; before that it only selects."""
    if data.workload not in WORKLOADS:
        raise HTTPException(422, f"unknown workload {data.workload!r}; expected one of {WORKLOADS}")
    if sim_state["engine"] is None:
        sim_state["workload"] = data.workload
        return {"status": "selected", **_run_status()}
    _start(data.workload, keep_settings=True)
    _emit("workload", workload=data.workload, controller=sim_state["controller"],
          controller_label=CONTROLLERS[sim_state["controller"]])
    return {"status": "restarted", **_run_status()}


@app.get("/api/status")
async def status():
    model_loaded = sim_state["model"] is not None
    return {"running": sim_state["running"], "tick": sim_state["tick"],
            **_run_status(),
            "workloads": WORKLOADS,
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


def _run_status() -> dict:
    """Workload, active controller, thresholds and cumulative counters since the last switch."""
    engine, base = sim_state["engine"], sim_state["switch_baseline"]
    since = None
    if engine is not None and base is not None:
        since = dict(ticks=engine.tick - base["tick"],
                     total_drops=engine.total_drops - base["total_drops"],
                     fabric_drops=engine.fabric_link_drops - base["fabric_drops"],
                     host_drops=engine.host_link_drops - base["host_drops"],
                     reroutes=sim_state["switch_reroutes"])
    return {"workload": sim_state["workload"],
            "controller": sim_state["controller"],
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
            if engine.tick >= sim_state["episode_end"]:
                _next_episode(engine.tick)
            ready = len(collector.node_features_history) >= MODEL["window_size"]
            # The forecast overlay is shown under every controller; only guarded_gnn routes on it.
            predictions = None
            if ready and sim_state["model"] is not None:
                predictions = _get_predictions(collector, sim_state["model"])
            reroutes, rerouter = [], sim_state["rerouter"]
            if ready and rerouter is not None:
                if sim_state["controller"] == "guarded_gnn":
                    rerouter.threshold = sim_state["threshold"]
                    signal, signal_name = predictions, "probability"
                else:
                    signal, signal_name = collector.edge_scores_history[-1], "score"
                events = rerouter.evaluate_and_reroute(sim_state["flows"], signal, engine, engine.tick)
                sim_state["total_reroutes"] += len(events)
                sim_state["switch_reroutes"] += len(events)
                reroutes = [_describe_reroute(evt, signal, signal_name, collector) for evt in events]
                sim_state["recent_events"].extend(dict(r, type="reroute") for r in reroutes)
            _tick_events(snapshot, collector, predictions)
            timeline, sim_state["pending_events"] = sim_state["pending_events"], []
            # Bound live telemetry history; offline collectors keep full scenarios.
            for name in ["node_features_history", "edge_features_history", "edge_labels_history",
                         "edge_scores_history", "snapshots"]:
                history = getattr(collector, name)
                del history[:-MODEL["window_size"]]
            msg = _build_update_message(snapshot, collector, predictions, reroutes, timeline)
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
            await websocket.send_json({"type": "topology", "data": _get_topology_data(sim_state["graph"]),
                                       "recent_events": list(sim_state["recent_events"])})
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
            elif cmd.get("action") == "workload":
                try:
                    await set_workload(WorkloadUpdate(workload=cmd.get("value")))
                except ValueError:
                    await websocket.send_json({"type": "error", "message": "Workload must be a string"})
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


def _reroute_trigger(path, signal, collector):
    """The fabric link on `path` with the highest guard signal: the link whose risk put the path
    over the alert threshold (the guard rates a path by its riskiest link). (link ID, value)."""
    best = (None, None)
    for u, v in zip(path, path[1:]):
        idx = collector.edge_to_idx.get((u, v))
        if idx is not None and (best[1] is None or float(signal[idx]) > best[1]):
            best = (_link_id(u, v), float(signal[idx]))
    return best


def _describe_reroute(evt, signal, signal_name, collector):
    """A guard reroute event as sent to viewers (a copy; the guard's own log is not modified)."""
    trigger, value = _reroute_trigger(evt["old_path"], signal, collector)
    return dict(id=next(_event_ids), tick=evt["tick"], flow_id=evt["flow_id"], src=evt["src"], dst=evt["dst"],
                reason=evt.get("reason", "guarded"), controller=sim_state["controller"],
                old_path=list(evt["old_path"]), new_path=list(evt["new_path"]),
                trigger_link=trigger, trigger_value=None if value is None else round(value, 4),
                trigger_signal=signal_name)


def _tick_events(snapshot, collector, predictions):
    """Queue this tick's transition events: fabric links that became congested (served_or_queued
    label 0 -> 1), links whose congestion cleared, and links whose GNN probability crossed the live
    alert threshold. At most one coalesced event of each type per tick."""
    labels = collector.edge_labels_history[-1]
    congested = {_link_id(u, v) for (u, v), label in zip(collector.switch_edges, labels) if label}
    onset, cleared = sim_state["congestion_tracker"].update(congested)
    if onset:
        merged, G = merge_directions(snapshot["links"]), collector.G
        details = []
        for link in sorted(onset):
            u, v = link.split("--")
            data, edge = merged[(u, v)], G.edges[u, v]
            details.append(dict(
                link=link, utilization=round(data["utilization"], 3),
                served=round(data["packets_out"] / edge["capacity"], 3), queue_depth=data["queue_depth"],
                queue_fraction=round(data["queue_depth"] / edge["buffer_size"], 3),
                score=round(congestion_score(data, edge["capacity"], edge["buffer_size"]), 3)))
        _emit("congestion", links=details)
    if cleared:
        _emit("congestion_cleared", links=sorted(cleared))
    if predictions is not None:
        threshold = sim_state["threshold"]
        above = {_link_id(u, v): float(p) for (u, v), p in zip(collector.switch_edges, predictions) if p > threshold}
        crossed, _ = sim_state["prediction_tracker"].update(set(above))
        if crossed:
            _emit("predicted", threshold=threshold,
                  links=[dict(link=link, probability=round(above[link], 4)) for link in sorted(crossed)])


def _build_update_message(snapshot, collector, predictions=None, reroute_events=None, events=None) -> dict:
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

    # Global metrics (cumulative). Fabric-link drops are what rerouting can affect; host-link
    # drops happen on the receiver's edge->host link, which no path choice avoids.
    engine = collector.engine
    metrics = {
        "tick": snapshot["tick"],
        "active_flows": snapshot["active_flows"],
        "total_packets": snapshot["total_packets"],
        "total_drops": snapshot["total_drops"],
        "fabric_drops": engine.fabric_link_drops,
        "host_drops": engine.host_link_drops,
        "drop_rate": round(snapshot["total_drops"] / max(1, snapshot["total_packets"]) * 100, 3),
        "total_reroutes": sim_state["total_reroutes"],
    }

    # Reroute events (see _describe_reroute) and the other timeline events of this tick.
    reroute_data = []
    for evt in reroute_events or []:
        reroute_data.append({
            "id": evt.get("id"),
            "tick": evt.get("tick", snapshot["tick"]),
            "flow_id": evt["flow_id"],
            "src": evt["src"],
            "dst": evt["dst"],
            "reason": evt.get("reason", "proactive"),
            "controller": evt.get("controller", sim_state["controller"]),
            "old_path": list(evt.get("old_path", [])),
            "new_path": list(evt["new_path"]),
            "trigger_link": evt.get("trigger_link"),
            "trigger_value": evt.get("trigger_value"),
            "trigger_signal": evt.get("trigger_signal"),
        })

    return {
        "type": "update",
        "links": links,
        "predictions": pred_data,
        "metrics": metrics,
        "reroute_events": reroute_data,
        "events": list(events or []),
        **_run_status(),
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
