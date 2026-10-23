"""Extended experiment harness for the NeuralFabric revision.

Adds, without changing the simulator's packet dynamics:
  * GuardX: GuardedRerouter with every admission rule switchable (leave-one-rule-out ablations)
    and with alpha / delta / kappa / M exposed for sweeps;
  * telemetry delay d_T (the controller and the forecaster see link state from t - d_T) and
    actuation delay d_A (a route change decided at t is installed d_A ticks later);
  * a forecast interval I (the forecaster runs every I ticks; its output is used from t + I on);
  * per-flow instrumentation: delivered / dropped packets and out-of-order deliveries;
  * an optional closed-loop transport variant (fixed-size flows, loss recovery, AIMD) for FCT.

With default arguments run_case_x reproduces research_benchmark.run_case exactly
(checked against research/benchmark_results_v4.json by verify_against_benchmark()).
"""
from paths import NF_ROOT, RESULTS, FIGS, SCRIPTS  # folder locations (see paths.py)
import copy, os, sys, time
from collections import Counter, defaultdict
from itertools import islice
from types import SimpleNamespace

import numpy as np
import networkx as nx

ROOT = os.environ.get('NF_ROOT', NF_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MODEL, REROUTING, SIMULATION  # noqa: E402
from simulation.network_engine import NetworkEngine  # noqa: E402
from simulation.telemetry import TelemetryCollector  # noqa: E402
from routing.rerouter import ProactiveRerouter, ReactiveRerouter  # noqa: E402

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']
INF = float('inf')


# ─────────────────────────────── controller ────────────────────────────────
class GuardX:
    """GuardedRerouter (routing/guarded.py) with switchable rules.

    threshold     alert threshold tau_m (flows whose path risk <= tau are not considered)
    max_changes   M  (None = unlimited)
    headroom      alpha (None = no capacity check)
    reserve       sequential capacity reservation (False = every decision sees the initial demand)
    min_improvement delta (0 = any candidate with risk <= current risk passes)
    stretch       max extra hops (None = unbounded)
    cooldown      kappa ticks between moves of the same flow (0 = none)
    """

    def __init__(self, graph, edge_to_idx, threshold, max_changes=4, headroom=.90, reserve=True,
                 min_improvement=.15, stretch=2, cooldown=None, k_paths=None, max_age=1,
                 rate_noise=0.0, rate_mode='exact', noise_seed=0):
        self.G, self.edge_to_idx = graph, edge_to_idx
        self.threshold = threshold
        self.max_changes = max_changes if max_changes is not None else 10 ** 9
        self.headroom, self.reserve = headroom, reserve
        self.min_improvement, self.stretch = min_improvement, stretch
        self.cooldown = REROUTING['reroute_cooldown_ticks'] if cooldown is None else cooldown
        self.k_paths = REROUTING['k_shortest_paths'] if k_paths is None else k_paths
        self.max_age = max_age
        self.last_reroute_tick, self.reroute_log = {}, []
        self.rejections = Counter()
        self.timing_ms = []
        self.rate_noise, self.rate_mode = rate_noise, rate_mode
        self._nrng = np.random.default_rng(noise_seed)

    def evaluate_and_reroute(self, flows, predictions, engine, tick, prediction_tick=None,
                             view=None, path_of=None, apply=None):
        """view: object with .link_states (defaults to engine); path_of(f): controller's belief of
        f's path (defaults to f.current_path); apply(f, path): installs a decision (defaults to
        setting f.current_path immediately)."""
        t0 = time.perf_counter()
        view = view or engine
        path_of = path_of or (lambda f: f.current_path)
        if prediction_tick is None:
            prediction_tick = tick
        expected = max(self.edge_to_idx.values(), default=-1) + 1
        p = np.asarray(predictions)
        if p.shape != (expected,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            self.rejections['invalid_prediction'] += 1
            return []
        if not 0 <= tick - prediction_tick <= self.max_age:
            self.rejections['stale_prediction'] += 1
            return []
        edges = lambda path: [engine._edge_key(u, v) for u, v in zip(path, path[1:])]

        def risk(path):
            return max((float(p[self.edge_to_idx[e]]) for e in edges(path) if e in self.edge_to_idx), default=0.)
        demand = Counter()
        active = [f for f in flows if f.active]
        # rate estimates used by the capacity check: exact, multiplicative log-normal noise
        # (fresh every tick), or rate-blind (every active flow assumed to send the mean active rate)
        if self.rate_mode == 'telemetry' and active:
            # receiver-side estimate: mean packets delivered per tick over the last five ticks;
            # flows without two ticks of history get the mean estimate of the others
            tel = getattr(self, 'telemetry_rates', {})
            known = [tel[f.flow_id] for f in active if f.flow_id in tel]
            fill = float(np.mean(known)) if known else float(np.mean([f.rate for f in active]))
            est = {f.flow_id: tel.get(f.flow_id, fill) for f in active}
        elif self.rate_mode == 'blind' and active:
            mean_rate = float(np.mean([f.rate for f in active]))
            est = {f.flow_id: mean_rate for f in active}
        elif self.rate_noise > 0:
            z = self._nrng.normal(0.0, self.rate_noise, len(active))
            est = {f.flow_id: f.rate * float(np.exp(zi - self.rate_noise ** 2 / 2)) for f, zi in zip(active, z)}
        else:
            est = {f.flow_id: f.rate for f in active}
        for f in active:
            for e in edges(path_of(f)):
                demand[e] += est[f.flow_id]
        for e, s in view.link_states.items():
            demand[e] = max(demand[e], s.packets_in)
        initial = Counter(demand)
        # per-tick link costs 1 + 10 * rho_e (identical values to the original closure, computed once)
        wmap = {e: 1 + 10 * float(p[i]) for e, i in self.edge_to_idx.items()}
        weight = lambda u, v, d: wmap.get((u, v), 1)
        memo = {}

        def cand(src, dst):
            # same weights within a tick -> same path sequence; generate lazily once per (src, dst)
            if (src, dst) not in memo:
                memo[(src, dst)] = ([], nx.shortest_simple_paths(self.G, src, dst, weight=weight))
            done, gen = memo[(src, dst)]
            i = 0
            while True:
                if i < len(done):
                    yield done[i]
                else:
                    try:
                        nxt = next(gen)
                    except StopIteration:
                        return
                    done.append(nxt)
                    yield nxt
                i += 1
        events = []
        for f in sorted(active, key=lambda f: (-risk(path_of(f)), f.flow_id)):
            if len(events) >= self.max_changes:
                break
            old = path_of(f)
            if risk(old) <= self.threshold:
                continue
            if tick - self.last_reroute_tick.get(f.flow_id, -10 ** 9) < self.cooldown:
                self.rejections['cooldown'] += 1
                continue
            old_edges = set(edges(old))

            candidates = cand(f.src, f.dst)
            chosen = None
            try:
                for path in islice(candidates, self.k_paths):
                    if path == old:
                        continue
                    if self.stretch is not None and len(path) > len(old) + self.stretch:
                        continue
                    if risk(old) - risk(path) < self.min_improvement:
                        continue
                    if self.headroom is not None:
                        dem = demand if self.reserve else initial
                        added = set(edges(path)) - old_edges
                        if not all(dem[e] + est[f.flow_id] + view.link_states[e].queue <=
                                   self.headroom * view.link_states[e].capacity for e in added):
                            continue
                    chosen = path
                    break
            except nx.NetworkXNoPath:
                pass
            if chosen is None:
                self.rejections['no_admissible_path'] += 1
                continue
            for e in old_edges:
                demand[e] -= est[f.flow_id]
            for e in edges(chosen):
                demand[e] += est[f.flow_id]
            if apply is None:
                f.current_path = chosen
            else:
                apply(f, chosen)
            self.last_reroute_tick[f.flow_id] = tick
            event = dict(tick=tick, flow_id=f.flow_id, old_path=old, new_path=chosen)
            events.append(event)
            self.reroute_log.append(event)
        self.timing_ms.append((time.perf_counter() - t0) * 1000)
        return events


class FlowBenderRerouter:
    """FlowBender-style baseline (Kabbani et al., CoNEXT 2014) in the tick model: a link marks ECN
    when its queue exceeds k_frac of the buffer; a flow whose current path contains a marking link for
    n_consec consecutive ticks is rehashed to a different equal-cost shortest path chosen at random,
    and its counter is reset. No capacity check, budget or risk signal."""

    def __init__(self, graph, k_frac=0.1, n_consec=2, seed=0):
        self.G, self.k_frac, self.n = graph, k_frac, n_consec
        self.rng = np.random.default_rng(seed)
        self.count, self.paths, self.reroute_log = {}, {}, []

    def evaluate_and_reroute(self, flows, engine, tick, apply=None, path_of=None):
        path_of = path_of or (lambda f: f.current_path)
        marking = {k for k, s in engine.link_states.items() if s.queue > self.k_frac * s.buffer_size}
        events = []
        for f in flows:
            if not f.active:
                continue
            path = path_of(f)
            if len(path) < 2:
                continue
            hit = any(engine._edge_key(u, v) in marking for u, v in zip(path, path[1:]))
            c = self.count.get(f.flow_id, 0) + 1 if hit else 0
            if c >= self.n:
                key = (f.src, f.dst)
                if key not in self.paths:
                    self.paths[key] = [list(p) for p in nx.all_shortest_paths(self.G, f.src, f.dst)]
                alts = [p for p in self.paths[key] if p != list(path)]
                if alts:
                    new = alts[int(self.rng.integers(len(alts)))]
                    if apply is None:
                        f.current_path = new
                    else:
                        apply(f, new)
                    ev = dict(tick=tick, flow_id=f.flow_id, old_path=path, new_path=new)
                    events.append(ev); self.reroute_log.append(ev)
                c = 0
            self.count[f.flow_id] = c
        return events


# ─────────────────────────────── engines ───────────────────────────────────
class InstrumentedEngine(NetworkEngine):
    """Same dynamics as NetworkEngine.step(); additionally records per-flow delivery,
    drops and out-of-order deliveries (a batch is out of order if a batch of the same
    flow injected later has already been delivered)."""

    def reset(self):
        super().reset()
        self.flow_delivered = Counter()
        self.flow_dropped = Counter()
        self.flow_reordered = Counter()
        self.flow_max_born = {}
        self.flow_last_delivery = {}
        self.delivered_per_tick = []

    def step(self):
        now = self.tick
        for state in self.link_states.values():
            state.prepare_tick()
        arrivals, self.pending = self.pending, []
        for f in self.active_flows:
            f.active = f.start_tick <= now < f.start_tick + f.duration
            if not f.active:
                continue
            self._fractional[f.flow_id] += f.rate
            count = int(self._fractional[f.flow_id])
            self._fractional[f.flow_id] -= count
            self.total_packets += count
            path = tuple(f.current_path)
            if len(path) < 2:
                self.total_drops += count
                self.unroutable_drops += count
                self.flow_dropped[f.flow_id] += count
            elif count:
                arrivals.append((count, (path, 0, now, f.flow_id)))
        for count, payload in arrivals:
            path, hop, born, flow_id = payload
            key = self._edge_key(path[hop], path[hop + 1])
            dropped = self.link_states[key].enqueue(count, payload)
            self.total_drops += dropped
            if dropped:
                self.flow_dropped[flow_id] += dropped
            if self._host_link[key]:
                self.host_link_drops += dropped
            else:
                self.fabric_link_drops += dropped
        delivered_now = 0
        for state in self.link_states.values():
            for count, payload in state.process_tick():
                path, hop, born, flow_id = payload
                if hop + 1 == len(path) - 1:
                    self.total_delivered += count
                    delivered_now += count
                    self.delivery_latency_ticks[now - born + 1] += count
                    self.flow_delivered[flow_id] += count
                    mb = self.flow_max_born.get(flow_id, -1)
                    if born < mb:
                        self.flow_reordered[flow_id] += count
                    else:
                        self.flow_max_born[flow_id] = born
                    self.flow_last_delivery[flow_id] = now
                else:
                    self.pending.append((count, (path, hop + 1, born, flow_id)))
        self.delivered_per_tick.append(delivered_now)
        self.tick += 1
        self.link_latency_history.extend(s.latency_us for s in self.link_states.values())
        self.link_util_history.extend(s.packets_out / s.capacity for s in self.link_states.values())
        return self._collect_snapshot()


class ClosedLoopEngine(NetworkEngine):
    """Closed-loop transport variant used only for the FCT experiment.

    Every flow has a fixed size S_f (the packets it offers in the open-loop run). The source
    sends at its current rate r_f: retransmissions first, then new packets. A dropped batch is
    detected by the sender one round-trip time later (RTT = 2 x path hops; acknowledgements are
    assumed loss-free), queued for retransmission, and triggers a multiplicative decrease
    r_f <- max(1, md * r_f) at most once per RTT; otherwise r_f grows by ai * R_f per tick up to
    the nominal rate R_f. A flow completes when all S_f packets have been delivered.
    Flow.rate always holds the current sending rate, so controllers see current demand.
    """

    def __init__(self, graph, ai=0.05, md=0.5, reorder_penalty=False):
        self.ai, self.md = ai, md
        # reorder_penalty: an out-of-order delivery is treated by the sender as a loss
        # (duplicate-ACK fast retransmit): after half an RTT the reordered packets are retransmitted
        # spuriously (they consume capacity but add no progress) and the rate is decreased
        # (at most once per RTT, shared with real losses). Every reordered batch is penalised,
        # i.e. the penalty is conservative (no duplicate-ACK threshold).
        self.reorder_penalty = reorder_penalty
        super().__init__(graph)

    def set_flows(self, flows):
        super().set_flows(flows)
        self.cl = {}
        self._start = {f.flow_id: f.start_tick for f in flows}
        for f in flows:
            hops = max(1, len(f.current_path) - 1)
            size = int(f.rate * f.duration + 1e-9)
            self.cl[f.flow_id] = SimpleNamespace(
                nominal=f.rate, size=size, sent_new=0, delivered=0, retx_ready=0, loss_events=[],
                last_decrease=-10 ** 9, done=size == 0, fct=None, retx_sent=0,
                spur_events=[], spur_ready=0, max_born=-1, reordered=0, spur_sent=0,
                ideal=int(np.ceil(size / f.rate)) - 1 + hops if size else 0)

    def reset(self):
        super().reset()
        self.retx_packets = 0
        self.spurious_packets = 0
        self.reordered_packets = 0

    def all_done(self):
        return all(s.done for s in self.cl.values())

    def step(self):
        now = self.tick
        for state in self.link_states.values():
            state.prepare_tick()
        arrivals, self.pending = self.pending, []
        for f in self.active_flows:
            st = self.cl[f.flow_id]
            if st.done or now < f.start_tick:
                f.active = False
                continue
            f.active = True
            detected, keep = 0, []
            for t_det, cnt in st.loss_events:
                if t_det <= now:
                    detected += cnt
                else:
                    keep.append((t_det, cnt))
            st.loss_events = keep
            spur, keep = 0, []
            for t_det, cnt in st.spur_events:
                if t_det <= now:
                    spur += cnt
                else:
                    keep.append((t_det, cnt))
            st.spur_events = keep
            st.spur_ready += spur
            detected_any = detected + spur
            rtt = max(2, 2 * (len(f.current_path) - 1))
            if detected_any:
                st.retx_ready += detected
                if now - st.last_decrease >= rtt:
                    f.rate = max(1.0, f.rate * self.md)
                    st.last_decrease = now
            elif f.rate < st.nominal:
                f.rate = min(st.nominal, f.rate + self.ai * st.nominal)
            self._fractional[f.flow_id] += f.rate
            budget = int(self._fractional[f.flow_id])
            self._fractional[f.flow_id] -= budget
            sp = min(budget, st.spur_ready)
            st.spur_ready -= sp
            st.spur_sent += sp
            self.spurious_packets += sp
            budget -= sp
            retx = min(budget, st.retx_ready)
            new = min(budget - retx, st.size - st.sent_new)
            st.retx_ready -= retx
            st.sent_new += new
            st.retx_sent += retx
            self.retx_packets += retx
            count = retx + new
            path = tuple(f.current_path)
            if sp and len(path) >= 2:
                self.total_packets += sp
                arrivals.append((sp, (path, 0, now, f.flow_id, True)))
            if count == 0:
                if not sp:
                    self._fractional[f.flow_id] = 0.0
                continue
            self.total_packets += count
            if len(path) < 2:
                self.total_drops += count
                self.unroutable_drops += count
                st.loss_events.append((now + rtt, count))
            else:
                arrivals.append((count, (path, 0, now, f.flow_id)))
        for count, payload in arrivals:
            path, hop, born, flow_id = payload[:4]
            spurious = len(payload) > 4
            key = self._edge_key(path[hop], path[hop + 1])
            dropped = self.link_states[key].enqueue(count, payload)
            self.total_drops += dropped
            if self._host_link[key]:
                self.host_link_drops += dropped
            else:
                self.fabric_link_drops += dropped
            if dropped and not spurious:
                self.cl[flow_id].loss_events.append((now + max(2, 2 * (len(path) - 1)), dropped))
        for state in self.link_states.values():
            for count, payload in state.process_tick():
                path, hop, born, flow_id = payload[:4]
                spurious = len(payload) > 4
                if hop + 1 == len(path) - 1:
                    self.total_delivered += count
                    self.delivery_latency_ticks[now - born + 1] += count
                    st = self.cl[flow_id]
                    if spurious:
                        continue
                    if born < st.max_born:
                        st.reordered += count
                        self.reordered_packets += count
                        if self.reorder_penalty and not st.done:
                            st.spur_events.append((now + max(1, len(path) - 1), count))
                    else:
                        st.max_born = born
                    st.delivered += count
                    if not st.done and st.delivered >= st.size:
                        st.done = True
                        st.fct = now + 1 - self._start[flow_id]
                else:
                    self.pending.append((count, (path, hop + 1, born, flow_id) + ((True,) if spurious else ())))
        self.tick += 1
        self.link_latency_history.extend(s.latency_us for s in self.link_states.values())
        self.link_util_history.extend(s.packets_out / s.capacity for s in self.link_states.values())
        return self._collect_snapshot()


# ─────────────────────────────── models ────────────────────────────────────
_MODELS = {}


def load_models(kinds=('gnn', 'lstm_only'), seeds=(0, 1, 2), threads=1):
    """Temperature-scaled checkpoints + their calibrated validation thresholds."""
    import torch
    torch.set_num_threads(threads)
    from model.gnn_model import CongestionPredictor, load_compatible_state_dict, check_checkpoint_schema
    from model.baselines import LSTMOnly
    from model.calibration import TemperatureScaled
    g = __import__('simulation.topology', fromlist=['build_fat_tree']).build_fat_tree()
    col = TelemetryCollector(g, NetworkEngine(g))
    edge_pairs = [(col.switch_to_idx[u], col.switch_to_idx[v]) for u, v in col.switch_edges]
    for kind in kinds:
        for k in seeds:
            stem = os.path.join(ROOT, 'checkpoints', 'best_model_v3' + ('' if k == 0 else f'_seed{k}'))
            path = stem + ('_lstm_only.pt' if kind == 'lstm_only' else '.pt')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            check_checkpoint_schema(ck)
            m = CongestionPredictor(num_edges=len(col.switch_edges), edge_pairs=edge_pairs) if kind == 'gnn' else LSTMOnly()
            load_compatible_state_dict(m, ck['model_state_dict'])
            m.eval()
            _MODELS[(kind, k)] = dict(model=TemperatureScaled(m, ck['temperature']).eval(),
                                      threshold=float(ck['alert_threshold']), temperature=float(ck['temperature']))
    return _MODELS


class _PlattBase:
    pass


def PlattScaled(raw_model, a, b):
    """Affine recalibration on the logit: p = sigmoid(a * logit(raw) + b)."""
    import torch
    import torch.nn as nn

    class _P(nn.Module):
        def __init__(self):
            super().__init__()
            self.m, self.a, self.b = raw_model, float(a), float(b)

        def forward(self, node_features, edge_features, edge_index=None):
            p = self.m(node_features, edge_features, edge_index).clamp(1e-6, 1 - 1e-6)
            return torch.sigmoid(self.a * torch.logit(p) + self.b)
    return _P().eval()


def predict_window(model, collector, end_offset=0):
    """Forecast from the W ticks ending d ticks ago (end_offset = d)."""
    import torch
    w = MODEL['window_size']
    nf, ef = collector.node_features_history, collector.edge_features_history
    stop = len(nf) - end_offset
    with torch.no_grad():
        return model(torch.tensor(np.stack(nf[stop - w:stop])).unsqueeze(0),
                     torch.tensor(np.stack(ef[stop - w:stop])).unsqueeze(0),
                     torch.tensor(collector.get_pyg_edge_index())).squeeze(0).numpy()


# ─────────────────────────────── one run ───────────────────────────────────
def _delayed_view(snapshot, capacity_of):
    return SimpleNamespace(link_states={k: SimpleNamespace(queue=v['queue_depth'], packets_in=v['packets_in'],
                                                           capacity=capacity_of[k])
                                        for k, v in snapshot['links'].items()})


def run_case_x(graph, flows, ticks, strategy, threshold=None, model=None, guard=None,
               d_T=0, d_A=0, interval=1, instrument=False, closed_loop=False, max_ticks=600,
               time_controller=False, rng_seed=0, reorder_penalty=False):
    """strategy: ecmp | reactive | unguarded | guarded (needs signal in guard['signal']).
    guard: dict(signal='score'|'persistence'|'model', **GuardX kwargs)."""
    if closed_loop:
        engine = ClosedLoopEngine(graph, reorder_penalty=reorder_penalty)
    elif instrument:
        engine = InstrumentedEngine(graph)
    else:
        engine = NetworkEngine(graph)
    flows = copy.deepcopy(flows)
    for f in flows:
        f.current_path = []
        f.active = False
    engine.set_flows(flows)
    c = TelemetryCollector(graph, engine)
    capacity_of = {k: s.capacity for k, s in engine.link_states.items()}
    guard = dict(guard or {})
    signal = guard.pop('signal', None)
    if strategy == 'reactive':
        controller = ReactiveRerouter(graph)
    elif strategy == 'flowbender':
        controller = FlowBenderRerouter(graph, seed=rng_seed, **guard)
    elif strategy == 'unguarded':
        controller = ProactiveRerouter(graph, c.edge_to_idx)
    elif strategy == 'guarded':
        controller = GuardX(graph, c.edge_to_idx, threshold=threshold, **guard)
    else:
        controller = None
    # actuation delay bookkeeping
    planned = {}                      # flow_id -> path the controller has decided
    pending = []                      # (install_tick, flow, path)

    def path_of(f):
        return planned.get(f.flow_id, f.current_path)

    def apply(f, path):
        planned[f.flow_id] = path
        if d_A == 0:
            f.current_path = path
        else:
            pending.append((engine.tick + d_A, f, path))
    last_pred, inflight = None, []
    inference = []
    rng = np.random.default_rng(rng_seed)
    static_risk = None
    tel_prev, tel_hist = {}, {}
    W = MODEL['window_size']
    n_ticks = max_ticks if closed_loop else ticks
    for _ in range(n_ticks):
        if pending:
            due = [x for x in pending if x[0] <= engine.tick]
            pending = [x for x in pending if x[0] > engine.tick]
            for _, f, path in due:
                f.current_path = path
        c.collect_tick(engine.step())
        if instrument and not closed_loop and isinstance(controller, GuardX) and controller.rate_mode == 'telemetry':
            cur = engine.flow_delivered
            for fid, v in cur.items():
                hist = tel_hist.setdefault(fid, [])
                hist.append(v - tel_prev.get(fid, 0))
                del hist[:-5]
                tel_prev[fid] = v
            controller.telemetry_rates = {fid: float(np.mean(h)) for fid, h in tel_hist.items() if len(h) >= 2}
        if closed_loop and engine.all_done():
            break
        if controller is None:
            continue
        if strategy == 'reactive':
            controller.evaluate_and_reroute(flows, engine, engine.tick)
            continue
        if strategy == 'flowbender':
            controller.evaluate_and_reroute(flows, engine, engine.tick)
            continue
        if engine.tick < W + d_T:
            continue
        if strategy == 'guarded' and signal in ('score', 'persistence'):
            hist = c.edge_scores_history if signal == 'score' else c.edge_labels_history
            p = hist[-1 - d_T]
            pred_tick = engine.tick
        elif strategy == 'guarded' and signal == 'constant':
            # signal-free: every link has the same risk, so every flow is eligible and the
            # candidate order is plain shortest paths; the capacity check alone decides
            p = np.ones(max(c.edge_to_idx.values()) + 1)
            pred_tick = engine.tick
        elif strategy == 'guarded' and signal in ('random', 'static'):
            # control signals without congestion information: i.i.d. uniform per link and tick,
            # or one uniform value per link drawn once per run
            if signal == 'random' or static_risk is None:
                draw = rng.random(max(c.edge_to_idx.values()) + 1)
                if signal == 'static':
                    static_risk = draw
            p = static_risk if signal == 'static' else draw
            pred_tick = engine.tick
        elif strategy == 'guarded' and signal in ('oracle_score', 'oracle_label'):
            # perfect H-tick look-ahead under the current routing: copy the simulator, advance it
            # H ticks without control, and read the true score / label of every fabric link
            eng2 = copy.deepcopy(engine, {id(graph): graph})
            c2 = TelemetryCollector(graph, eng2)
            for _ in range(MODEL['prediction_horizon']):
                c2.collect_tick(eng2.step())
            hist = c2.edge_scores_history if signal == 'oracle_score' else c2.edge_labels_history
            p = np.asarray(hist[-1], dtype=float)
            pred_tick = engine.tick
        else:
            # model signal (guarded model or unguarded GNN)
            if interval == 1:
                s = time.perf_counter()
                p = predict_window(model, c, end_offset=d_T)
                inference.append((time.perf_counter() - s) * 1000)
                pred_tick = engine.tick
            else:
                # invoked every `interval` ticks; each output becomes usable `interval` ticks
                # after its invocation (the invocation's time budget); the latest usable
                # forecast is used in between.
                if (engine.tick - W - d_T) % interval == 0:
                    s = time.perf_counter()
                    newp = predict_window(model, c, end_offset=d_T)
                    inference.append((time.perf_counter() - s) * 1000)
                    inflight.append((engine.tick + interval, newp))
                while inflight and inflight[0][0] <= engine.tick:
                    last_pred = inflight.pop(0)[1]
                if last_pred is None:
                    continue
                p, pred_tick = last_pred, engine.tick
        if strategy == 'unguarded':
            controller.evaluate_and_reroute(flows, p, engine, engine.tick)
            continue
        view = _delayed_view(c.snapshots[-1 - d_T], capacity_of) if d_T else None
        controller.evaluate_and_reroute(flows, p, engine, engine.tick, prediction_tick=pred_tick,
                                        view=view, path_of=path_of, apply=apply)
    result = engine.metrics()
    result['ticks_run'] = engine.tick
    result['reroutes'] = len(controller.reroute_log) if controller is not None else 0
    if inference:
        result['inference_ms_p50'] = float(np.median(inference))
        result['inference_ms_p99'] = float(np.percentile(inference, 99))
    if isinstance(controller, GuardX):
        result['rejections'] = dict(controller.rejections)
        if time_controller:
            result['controller_ms'] = controller.timing_ms
            result['inference_ms'] = inference
    if instrument and not closed_loop:
        dl = engine.flow_delivered
        tot_del = sum(dl.values())
        result['reordered_packets'] = int(sum(engine.flow_reordered.values()))
        result['reordered_fraction'] = result['reordered_packets'] / tot_del if tot_del else 0.0
        result['flows_with_reordering'] = int(sum(1 for v in engine.flow_reordered.values() if v))
        result['n_flows'] = len(flows)
        loss = [engine.flow_dropped[f.flow_id] / max(1, engine.flow_dropped[f.flow_id] + dl[f.flow_id]) for f in flows]
        result['flow_loss_p50'] = float(np.median(loss))
        result['flow_loss_p90'] = float(np.percentile(loss, 90))
        result['flow_loss_mean'] = float(np.mean(loss))
    if closed_loop:
        cl = engine.cl
        done = [s for s in cl.values() if s.done and s.size]
        fct = [s.fct for s in done]
        slow = [s.fct / s.ideal for s in done if s.ideal]
        result.update(n_flows=len(flows), completed=len(done), incomplete=sum(1 for s in cl.values() if not s.done),
                      fct_mean=float(np.mean(fct)) if fct else None,
                      fct_p50=float(np.median(fct)) if fct else None,
                      fct_p99=float(np.percentile(fct, 99)) if fct else None,
                      slowdown_mean=float(np.mean(slow)) if slow else None,
                      slowdown_p50=float(np.median(slow)) if slow else None,
                      slowdown_p99=float(np.percentile(slow, 99)) if slow else None,
                      makespan=engine.tick, retx_packets=engine.retx_packets,
                      spurious_packets=engine.spurious_packets, cl_reordered_packets=engine.reordered_packets,
                      offered_unique=int(sum(s.size for s in cl.values())),
                      per_flow=[(f.flow_id, f.flow_type, cl[f.flow_id].size, cl[f.flow_id].ideal, cl[f.flow_id].fct)
                                for f in flows])
    return result
